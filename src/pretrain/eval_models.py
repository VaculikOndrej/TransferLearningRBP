from statistics import mean


def get_highest_average(aucs):
    avg_auc_dict = {model: mean(scores) for model, scores in aucs.items()}
    max_auc = max(avg_auc_dict.values())
    max_model = [model for model in avg_auc_dict.keys() if avg_auc_dict[model] == max_auc][0]
    return max_model, max_auc

def get_lowest_average(aucs):
    avg_auc_dict = {model: mean(scores) for model, scores in aucs.items()}
    min_auc = min(avg_auc_dict.values())
    min_model = [model for model in avg_auc_dict.keys() if avg_auc_dict[model] == min_auc][0]
    return min_model, min_auc