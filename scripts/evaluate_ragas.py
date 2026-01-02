"""RAGAS Evaluation Script for RAG System - Optimized for Gemini & RAGAS 0.2+"""
import os
import sys
import json
import math
from datetime import datetime
from typing import List, Dict

# 1. Standart ve Üçüncü Parti Importlar (Pylint C0413 hatasını önlemek için en üstte)
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

# RAGAS Metrikleri ve Wrapper'lar (Pylint E0611 hataları için disable eklendi)
from ragas.llms import LangchainLLMWrapper # pylint: disable=no-name-in-module
from ragas.embeddings import LangchainEmbeddingsWrapper # pylint: disable=no-name-in-module
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# LangChain Gemini Entegrasyonu
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 2. Yerel Modül Yolu ve Importları
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.multi_series_service import MultiSeriesService
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics and Gemini API."""
    
    def __init__(self, test_set_path: str = "data/test/test_set.json"):
        self.test_set_path = test_set_path
        self.service = MultiSeriesService()
        self.results_dir = "data/test/results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Gemini API Kontrolü
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY eksik! Lütfen .env dosyanızı kontrol edin.")

        # Gemini Modellerini İlklendir
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash-lite",
            temperature=0.0
        )
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        
        # RAGAS için Wrapper'lar
        self.ragas_llm = LangchainLLMWrapper(self.gemini_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.gemini_embeddings)

    def load_test_set(self) -> List[Dict]:
        """Load test set from JSON file."""
        with open(self.test_set_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_cases']

    def run_query(self, test_case: Dict) -> Dict:
        """Run a single query through the RAG system."""
        question = test_case['question']
        series = test_case.get('series', 'stranger_things')
        
        logger.info(f"Processing: {question[:50]}...")
        
        try:
            result = self.service.query_single_series(
                series_name=series,
                query=question
            )
            # Sources listesinden content'leri çıkar
            contexts = [src.get('content', '') if isinstance(src, dict) else getattr(src, 'content', '') 
                        for src in result.sources]
            
            return {
                'question': question,
                'answer': result.answer,
                'contexts': contexts,
                'ground_truth': test_case['ground_truth']
            }
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'contexts': [],
                'ground_truth': test_case['ground_truth']
            }

    def evaluate_test_set(self) -> Dict:
        """Evaluate entire test set using Gemini-backed RAGAS metrics."""
        test_cases = self.load_test_set()
        logger.info(f"Loaded {len(test_cases)} test cases")
        
        eval_data = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"[{i}/{len(test_cases)}] Running query...")
            eval_data.append(self.run_query(test_case))
        
        # RAGAS Formatına Dönüştür
        dataset = Dataset.from_dict({
            'question': [d['question'] for d in eval_data],
            'answer': [d['answer'] for d in eval_data],
            'contexts': [d['contexts'] for d in eval_data],
            'ground_truth': [d['ground_truth'] for d in eval_data]
        })
        
        # Metrikleri Gemini ile Yapılandır
        metrics = [
            Faithfulness(llm=self.ragas_llm),
            AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings),
            ContextPrecision(llm=self.ragas_llm),
            ContextRecall(llm=self.ragas_llm)
        ]

        logger.info("Running RAGAS evaluation with Gemini API...")
        
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )
        
        return {
            'ragas_scores': results,
            'eval_data': eval_data,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_set_size': len(test_cases),
                'test_set_path': self.test_set_path,
                'model_type': 'gemini-cloud'
            }
        }

    def save_results(self, results: Dict, output_name: str = None):
        """Save evaluation results with NaN handling."""
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"ragas_eval_{timestamp}.json"
        
        output_path = os.path.join(self.results_dir, output_name)
        
        # RAGAS 0.4.2'de sonuç formatı farklı - dict olarak al
        ragas_result = results['ragas_scores']
        
        # Eğer dict ise direkt kullan, değilse to_pandas() veya attributes kullan
        if isinstance(ragas_result, dict):
            raw_scores = ragas_result
        elif hasattr(ragas_result, 'to_pandas'):
            # DataFrame ise dict'e çevir
            raw_scores = ragas_result.to_pandas().to_dict('records')[0]
        elif hasattr(ragas_result, '__dict__'):
            # Object ise attributes'e bak
            raw_scores = {k: v for k, v in ragas_result.__dict__.items() if not k.startswith('_')}
        else:
            raw_scores = {}
        
        cleaned_scores = {
            k: (0.0 if (isinstance(v, float) and math.isnan(v)) else v) 
            for k, v in raw_scores.items()
        }
        
        serializable_results = {
            'metadata': results['metadata'],
            'scores': cleaned_scores,
            'detailed_results': results['eval_data']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path

    def print_summary(self, results: Dict):
        """Print evaluation summary."""
        ragas_result = results['ragas_scores']
        
        # Skorları çıkar
        if isinstance(ragas_result, dict):
            scores = ragas_result
        elif hasattr(ragas_result, 'to_pandas'):
            scores = ragas_result.to_pandas().to_dict('records')[0]
        elif hasattr(ragas_result, '__dict__'):
            scores = {k: v for k, v in ragas_result.__dict__.items() if not k.startswith('_')}
        else:
            scores = {}
        
        print("\n" + "="*60)
        print("GEMINI-BASED RAGAS EVALUATION RESULTS")
        print("="*60)
        for metric, val in scores.items():
            if isinstance(val, (int, float)):
                val_print = 0.0 if math.isnan(val) else val
                print(f"{metric.ljust(25)}: {val_print:.4f}")
            else:
                print(f"{metric.ljust(25)}: {val}")
        print("="*60 + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate RAG system using Gemini & RAGAS')
    parser.add_argument('--test-set', default='data/test/test_set.json', help='Test set path')
    args = parser.parse_args()
    
    evaluator = RAGASEvaluator(test_set_path=args.test_set)
    
    try:
        results = evaluator.evaluate_test_set()
        output_path = evaluator.save_results(results)
        evaluator.print_summary(results)
        logger.info(f"Evaluation complete! Saved to: {output_path}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()