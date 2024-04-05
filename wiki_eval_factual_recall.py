import argparse
from faaf.experiments import FaafFactVerification, PromptFactVerification
from faaf.utils import time_function

@time_function
def run_single_experiment(experiment_class, **parameters):
    experiment = experiment_class(**parameters)
    print(10 * ' - - --')
    print(f'Running {experiment.exp_name} on WikiEval...')
    experiment.run_experiment()
    print(f'\n--- Score Summary for {experiment.exp_name} ---')
    print(experiment.get_scores())
    #experiment.save_scores()
    #experiment.save_results()

    return

def str2llm(v):
    '''
    Parses the llm name from user input with some flexibility.
    '''
    if isinstance(v, bool):
       return v
    if 'gpt' in v.lower() and '3' in v.lower():
        return "gpt-3.5-turbo-0125"
    elif 'gpt' in v.lower() and '4' in v.lower():
        return "gpt-4-0125-preview"
    elif 'calude' in v.lower() and 'opus' in v.lower():
        return "claude-3-opus-20240229"
    elif 'calude' in v.lower() and 'sonnet' in v.lower():
        return "claude-3-sonnet-20240229"
    elif 'mistral' in v.lower():
        return "mistral-large-latest"
    else:
        raise argparse.ArgumentTypeError(
            '''Choose one of the accepetd llm models:
            gpt-3.5-turbo-0125,
            gpt-4-0125-preview,
            mistral-large-latest,
            claude-3-opus-20240229,
            claude-3-sonnet-20240229 '''
            )
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():
    # parse num_threads for the different models
    parser = argparse.ArgumentParser(description="Parse user inputs")
    parser.add_argument('--oai_num_threads', type=int, help='Number of threads to use', default=1)
    parser.add_argument('--anthropic_num_threads', type=int, help='Number of threads to use', default=1)
    parser.add_argument('--mistral_num_threads', type=int, help='Number of threads to use', default=1)
    parser.add_argument('--auto', type=str2bool, default=False, help='Run all models and all verification methods - reproduce paper.')
    parser.add_argument('--llm', type=str2llm, default='gpt-3.5-turbo-0125', help='Choose llm to run.')

    args = parser.parse_args()

    # Mistral is excluded to to high validation error rates
    # owing to badly formated responses for function calling..
    llm2num_threads_mapping = {
        "gpt-3.5-turbo-0125": args.oai_num_threads,
        "gpt-4-0125-preview": args.oai_num_threads,
        "claude-3-opus-20240229": args.anthropic_num_threads,
        "claude-3-sonnet-20240229": args.anthropic_num_threads,
        "mistral-large-latest": args.mistral_num_threads
        }

    if args.auto:
        experiments = []
        llms_to_reproduce_paper = [
            "gpt-3.5-turbo-0125", 
            "gpt-4-0125-preview",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229"]

        for llm_name in llms_to_reproduce_paper:
            experiments.append(
                {
                    f"prompt-{llm_name}":{
                        "llm_name": llm_name,
                        "num_threads": llm2num_threads_mapping[llm_name]
                        },
                    f"faaf-tf-{llm_name}":{
                        "llm_name": llm_name,
                        "num_threads": llm2num_threads_mapping[llm_name],
                        "accept_not_clear_response": False,
                        "citation": False
                        },
                    f"faaf-tfn-{llm_name}":{
                        "llm_name": llm_name,
                        "num_threads": llm2num_threads_mapping[llm_name],
                        "accept_not_clear_response": True,
                        "citation": False
                        },
                    f"faaf-cit-{llm_name}":{
                        "llm_name": llm_name,
                        "num_threads": llm2num_threads_mapping[llm_name],
                        "accept_not_clear_response": False,
                        "citation": True
                        },
                    f"faaf-tfn-cit-{llm_name}":{
                        "llm_name": llm_name,
                        "num_threads": llm2num_threads_mapping[llm_name],
                        "accept_not_clear_response": True,
                        "citation": True
                        }
                }

            )
        # flatten experiments list to a single dict 
        experiments = {k: v for d in experiments for k, v in d.items()}
    else:
        experiments = {
            f"prompt-{args.llm}":{
                "llm_name": args.llm,
                "num_threads": llm2num_threads_mapping[args.llm]
                },
            f"faaf-tf-{args.llm}":{
                "llm_name": args.llm,
                "num_threads": llm2num_threads_mapping[args.llm],
                "accept_not_clear_response": False,
                "citation": False
                },
            f"faaf-tfn-{args.llm}":{
                "llm_name": args.llm,
                "num_threads": llm2num_threads_mapping[args.llm],
                "accept_not_clear_response": True,
                "citation": False
                },
            f"faaf-cit-{args.llm}":{
                "llm_name": args.llm,
                "num_threads": llm2num_threads_mapping[args.llm],
                "accept_not_clear_response": False,
                "citation": True
                },
            f"faaf-tfn-cit-{args.llm}":{
                "llm_name": args.llm,
                "num_threads": llm2num_threads_mapping[args.llm],
                "accept_not_clear_response": True,
                "citation": True
                }
            }

    for exp, params in experiments.items():
        print(f"\n\n\nrunning exp ---------------> {exp}")
        print(f"exp params ---------------> {params}")
        if 'faaf' in exp:
            run_single_experiment(
                experiment_class=FaafFactVerification,
                **params
                )
            
        elif 'prompt' in exp:
            run_single_experiment(
                experiment_class=PromptFactVerification,
                **params
                )


if __name__ == "__main__":
    main()