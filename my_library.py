def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def test_load():
  return 'loaded'

def test_it():
  return 'loaded'

def cond_prob(table, evidence, evidence_val, target1, target_val):
  t_subset = up_table_subset(table, target1, 'equals', target_val)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_val else 0 for v in e_list])/len(e_list)
  return p_b_a

def cond_probs_product(table, evidence_val, target1, target_val):
  cond_prob_list = []
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_val)
  for a, b in evidence_complete:
    prob = cond_prob(table, a, b, target_column, target_val)
    cond_prob_list += [prob]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob (table, target1, target_val):
  t_list = up_get_column(table, target1)
  p_a = sum([1 if v==target_val else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target1):
  neg = cond_probs_product(table, evidence_row, target1, 0) * prior_prob(table, target1, 0)
  pos = cond_probs_product(table, evidence_row, target1, 1) * prior_prob(table, target1, 1)
  neg,pos = compute_probs(neg, pos)
  return [neg,pos]

