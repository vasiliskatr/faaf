from abc import ABC, abstractmethod
from faaf.utils import llm_init, load_test_data_hugging_face
import concurrent.futures
import pandas as pd
from faaf.faaf_verification import fact_check_faaf_
from faaf.prompt_fact_verification import fact_check_prompt
from faaf.metrics import data_prep_for_scoring, error_rate_, f1_micro_
import os

class Experiment(ABC):
    def __init__(self, llm_name: str, num_threads: int):
        assert isinstance(num_threads, int) and num_threads > 0
        self.num_threads = num_threads
        self.llm_name = llm_name
        self.norm_llm_name = self.llm_name.replace("-", "").replace(".", "")
        self.client = llm_init(llm_name)
        self.test_data = load_test_data_hugging_face()
        self.scores = {}

        #for testing
        self.test_data = self.test_data.head(2)

    @abstractmethod
    def task_to_run(self, row, **kwargs):
        """
        The specific task each subclass will execute in parallel.
        To be implemented by subclasses.
        """
        pass

    def run_parallel(self, **kwargs):
        """
        Executes tasks in parallel on the test_data using multiple threads.
        This method uses a ThreadPoolExecutor to concurrently execute the `task_to_run` 
        method on each row of the `test_data` attribute of the instance.
        It requires the 'answer_col' keyword argument which points to the answer type of WikiEval
        to be specified for parallel execution. 
        The results from each task are aggregated into a pandas DataFrame and returned.
        """
        assert "answer_col" in kwargs, "answer_col not provided for parallel execution!"
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Map of the function task_to_run to each row
            futures = [executor.submit(self.task_to_run, row, **kwargs) for _, row in self.test_data.iterrows()]
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        results_df = pd.DataFrame(results)

        return results_df

    @abstractmethod
    def run_experiment(self):
        """To be implemented by subclasses."""
        pass

    def get_results(self):
        """If called after run_experiment() method, it returns all the detailed responses from the experiments."""
        return self.test_data
    
    def save_results(self):
        """Save results as a .csv in exp_results/"""
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'exp_results')
        # Check if the data directory exists, create it if it doesn't
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.test_data.to_csv(os.path.join(results_dir, f"{self.exp_name}_results.csv"), index=False)
        return 

    def get_scores(self):
        """Retrun scores of the experiments"""
        return pd.DataFrame(self.scores).T
    
    def save_scores(self):
        """Save scores as a .csv in exp_scores/"""
        scores_dir = os.path.join(os.path.dirname(__file__), '..', 'exp_scores')

        # Check if the data directory exists, create it if it doesn't
        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)

        scores_df = self.get_scores()
        scores_df.to_csv(os.path.join(scores_dir, f"{self.exp_name}_scores.csv"), index=False)

        return

class PromptFactVerification(Experiment):
    def __init__(self, llm_name, num_threads):
        super().__init__(llm_name, num_threads)
        self.exp_name = f"prompt_{self.norm_llm_name}"

    def task_to_run(self, row: dict, **kwargs):
        """
        Run fact checking task using the prompt-based verification, the specified row data and additional keyword arguments.
        """
        return fact_check_prompt(
            client=self.client, 
            gt_facts=row['gt_facts'], 
            context=row[kwargs["answer_col"]],
            unique_id=row['unique_id'],
            llm_name=self.llm_name
            )
    
    def run_experiment(self):
        """
        This function iterates through different answer types and runs fact checking in parallel.
        It merges the verification results with the main dataframe and calculates evaluation metrics
        such as F1 score and error rate.
        Finally, it stores the evaluation results in the 'scores' attribute of the class instance.

        All LLM calls are executed without retries. Failed LLM responses (where neither
        True or False can be matched in the LLM's response) are excluded from scoring.
        This way, the LLM's fact checking ability and output formating ablitly can be
        assessed independently.
        """
        for answer_col in ['answer', 'ungrounded_answer', 'poor_answer']:

            results_df = self.run_parallel(
                answer_col=answer_col
                )
            
            # append the llm name and answer type to the results column names except the "unique_id" column
            new_column_names = {
                col: f"{col}_{self.norm_llm_name}_{answer_col}"
                for col in results_df.columns if col != "unique_id"
                }
            results_df = results_df.rename(columns=new_column_names)
            verified_facts_col_name = f"prompt_evaluated_facts_{self.norm_llm_name}_{answer_col}"
           
            self.test_data = pd.merge(self.test_data, results_df, on='unique_id', how='left')

            # In this step we will discard any failed LLM responses.
            (
                flattened_human_annotated_facts,
                flattened_pred_facts, 
                failed_facts
                ) = data_prep_for_scoring(
                    df=self.test_data,
                    human_annotated_facts_col=f"human_evaluated_facts_{answer_col}",
                    pred_facts_col=verified_facts_col_name,
                    is_valid_col=None
                )
            
            er = error_rate_(
                human_annotated_facts_dict=flattened_human_annotated_facts,
                pred_facts_dict=flattened_pred_facts
                )
             
            f1  = f1_micro_(
                human_annotated_facts_dict=flattened_human_annotated_facts,
                pred_facts_dict=flattened_pred_facts
                )
            print(f"F1_micro: {f1}")
            print(f"error_rate: {er}")
             
            self.scores[verified_facts_col_name] = {
                'failed_facts':failed_facts,
                'num_failed_facts':len(failed_facts),
                'f1micro':f1,
                'ER': er
                }
        return

class FaafFactVerification(Experiment):
    def __init__(self, llm_name, num_threads, **kwargs):
        super().__init__(llm_name, num_threads)
        assert 'accept_not_clear_response' in kwargs, "accept_not_clear_response field is required"
        assert 'citation' in kwargs, "citation field is required"
        assert isinstance(kwargs['accept_not_clear_response'], bool), "accept_not_clear_response must be boolean"
        assert isinstance(kwargs['citation'], bool), "citation must be boolean"
        self.accept_not_clear_response = kwargs["accept_not_clear_response"]
        self.citation = kwargs["citation"]
        # tfn is abreviation for {true/false/not_clear}
        self.exp_name = f"faaf{'_tfn' if self.accept_not_clear_response else ''}{'_cit' if self.citation else ''}_{self.norm_llm_name}"

    def task_to_run(self, row: dict, **kwargs):
        """
        Run fact checking task using FaaF the specified row data and additional keyword arguments.
        """
        return fact_check_faaf_(
            client=self.client, 
            gt_facts=row['gt_facts'], 
            tfn=self.accept_not_clear_response,
            citation=self.citation,
            unique_id=row['unique_id'],
            context=row[kwargs["answer_col"]],
            llm_name=self.llm_name
            )
        
    def run_experiment(self):
        """
        This function iterates through different answer types and runs fact checking in parallel.
        It merges the verification results with the main dataframe and calculates evaluation metrics
        such as F1 score and error rate.
        Finally, it stores the evaluation results in the 'scores' attribute of the class instance.

        All LLM calls are executed without retries. Failed LLM responses due to bad formating
        are replaced by mock responses and excluded from scoring. This way, the LLM's fact checking ability
        and output formating ablitly can be assessed independently.
        """
        for answer_col in ['answer', 'ungrounded_answer', 'poor_answer']:

            results_df = self.run_parallel(
                answer_col=answer_col
                )
            
            # append the llm name and answer type to the results column names except the "unique_id" column
            new_column_names = {
                col: f"faaf{'_tfn' if self.accept_not_clear_response else ''}{'_cit' if self.citation else ''}_{col}_{self.norm_llm_name}_{answer_col}"
                for col in results_df.columns if col != "unique_id"
                }
            results_df = results_df.rename(columns=new_column_names)

            verified_facts_col_name = f"faaf{'_tfn' if self.accept_not_clear_response else ''}{'_cit' if self.citation else ''}_evaluated_facts_{self.norm_llm_name}_{answer_col}"
            valid_llm_response_col_name = f"faaf{'_tfn' if self.accept_not_clear_response else ''}{'_cit' if self.citation else ''}_valid_response_{self.norm_llm_name}_{answer_col}"             
            # merge LM verification results with main dataframe
            self.test_data = pd.merge(self.test_data, results_df, on='unique_id', how='left')

            # In this step we will discard any failed LLM responses.
            (
                flattened_human_annotated_facts,
                flattened_pred_facts, 
                failed_facts
                ) = data_prep_for_scoring(
                    df=self.test_data,
                    human_annotated_facts_col=f"human_evaluated_facts_{answer_col}",
                    pred_facts_col=verified_facts_col_name,
                    is_valid_col=valid_llm_response_col_name
                )
            
            er = error_rate_(
                human_annotated_facts_dict=flattened_human_annotated_facts,
                pred_facts_dict=flattened_pred_facts
                )
             
            f1  = f1_micro_(
                human_annotated_facts_dict=flattened_human_annotated_facts,
                pred_facts_dict=flattened_pred_facts
                )
            print(f"F1_micro: {f1}")
            print(f"error_rate: {er}")
             
            self.scores[verified_facts_col_name] = {
                'failed_facts':failed_facts,
                'num_failed_facts':len(failed_facts),
                'f1micro':f1,
                'ER': er
                }
        return 
