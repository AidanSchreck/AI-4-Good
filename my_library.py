def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def test_load():
  return 'loaded'

def test_it():
  return 'loaded'


def cond_prob(table, evidence, evidence_val, target, target_val):
  t_subset = up_table_subset(table, target, 'equals', target_val)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_val else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table, evidence_val, target, target_val):
  cond_prob_list = []
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_val)
  for a, b in evidence_complete:
    prob = cond_prob(table, a, b, target, target_val)
    cond_prob_list += [prob]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob (table, target, target_val):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_val else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  neg = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)
  pos = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)
  neg,pos = compute_probs(neg, pos)
  return [neg,pos]

def metrics (pred_label_list):
  assert isinstance(pred_label_list, list), f'Expecting parameter to be a list but instead is {type(pred_label_list)}'
  p = []
  for p in pred_label_list:
    assert isinstance(p, list), f'Expecting parameter to be a list of lists but instead is {type(pred_label_list)}'
    assert len(p)==2, f'Parameter is not a zipped list'
    
    for x in p:
        assert isinstance(x, int), f'Each value in the pair is not an int'
        assert x>=0, f'Some value in the pair is < 0'
  
  accuracy = sum(p==a for p, a in pred_label_list)/len(pred_label_list)
  tn = sum([1 if pair==[0,0] else 0 for pair in pred_label_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in pred_label_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in pred_label_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in pred_label_list])
  precision = tp/(tp + fp) if (tp + fp) != 0 else 0
  recall = tp/(tp + fn) if (tp + fn) !=0 else 0
  f1 = 2 * ((precision * recall)/(precision + recall)) if (precision + recall) !=0 else 0
  return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}

