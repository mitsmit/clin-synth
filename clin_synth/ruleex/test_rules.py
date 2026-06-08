import pandas as pd
import ast

def parse_frozenset_string(set_string):
    """
    Cleans a string like "frozenset({'a', 'b'})" down to "{'a', 'b'}"
    so ast.literal_eval can safely parse it.
    """
    if set_string.startswith("frozenset("):
        set_string = set_string[10:-1]
    elif set_string.startswith("set("):
        set_string = set_string[4:-1]
    return set(ast.literal_eval(set_string))

def bin_synthetic_row(row):
    """
    Transforms raw synthetic numeric and text columns into binned feature strings.
    """
    binned_features = set()
    
    # 1. Map Categorical values straight to feature strings
    for col in ['race', 'gender', 'age', 'A1Cresult', 'metformin', 'insulin', 'readmitted']:
        if col in row and pd.notna(row[col]):
            binned_features.add(f"{col}_{row[col]}")
            
    # 2. Re-apply clinical binning boundaries
    if 'time_in_hospital' in row:
        t = row['time_in_hospital']
        if t <= 2: binned_features.add('time_in_hospital_short_stay')
        elif t <= 5: binned_features.add('time_in_hospital_med_stay')
        else: binned_features.add('time_in_hospital_long_stay')
        
    if 'num_lab_procedures' in row:
        l = row['num_lab_procedures']
        if l < 30: binned_features.add('num_lab_procedures_low_labs')
        elif l < 60: binned_features.add('num_lab_procedures_med_labs')
        else: binned_features.add('num_lab_procedures_high_labs')
        
    if 'num_medications' in row:
        m = row['num_medications']
        if m < 12: binned_features.add('num_medications_low_meds')
        elif m < 22: binned_features.add('num_medications_med_meds')
        else: binned_features.add('num_medications_high_meds')
        
    if 'number_outpatient' in row:
        binned_features.add('number_outpatient_no_outpatient' if row['number_outpatient'] == 0 else 'number_outpatient_has_outpatient')
        
    if 'number_emergency' in row:
        binned_features.add('number_emergency_no_emergency' if row['number_emergency'] == 0 else 'number_emergency_has_emergency')
        
    if 'number_inpatient' in row:
        binned_features.add('number_inpatient_no_prior_inpatient' if row['number_inpatient'] == 0 else 'number_inpatient_has_prior_inpatient')
        
    return binned_features

def generate_rule_coverage_report(synthetic_data_path, rules_path):
    """
    Validates synthetic data against auto-mined hard rules and builds an aggregated 
    explainability summary counting how many times each rule was triggered and passed.
    """
    syn_df = pd.read_csv(synthetic_data_path)
    rules_df = pd.read_csv(rules_path)
    
    rules_df['antecedents'] = rules_df['antecedents'].apply(parse_frozenset_string)
    rules_df['consequents'] = rules_df['consequents'].apply(parse_frozenset_string)
    
    # Initialize a dictionary to track aggregated counts for each rule index
    # Schema: { rule_index: { "description": str, "times_triggered": int, "times_passed": int } }
    rule_metrics = {}
    
    for idx, rule in rules_df.iterrows():
        antecedent_list = list(rule['antecedents'])
        consequent_list = list(rule['consequents'])
        rule_description = f"IF {antecedent_list} -> THEN {consequent_list}"
        
        rule_metrics[idx] = {
            "rule_id": f"RULE_{idx:03d}",
            "rule_description": rule_description,
            "times_triggered": 0,
            "times_passed": 0,
            "times_failed": 0
        }
        
    passed_rows = []
    failed_rows_with_feedback = []
    
    print(f"Analyzing {len(syn_df)} synthetic rows against {len(rules_df)} rules...")
    
    for idx, raw_row in syn_df.iterrows():
        row_features = bin_synthetic_row(raw_row.to_dict())
        row_failed = False
        violated_explanations = []
        
        for rule_idx, rule in rules_df.iterrows():
            antecedent = rule['antecedents']
            consequent = rule['consequents']
            
            # Did this specific patient trigger the IF condition?
            if antecedent.issubset(row_features):
                rule_metrics[rule_idx]["times_triggered"] += 1
                
                # Did they pass the THEN condition?
                if consequent.issubset(row_features):
                    rule_metrics[rule_idx]["times_passed"] += 1
                else:
                    rule_metrics[rule_idx]["times_failed"] += 1
                    row_failed = True
                    violated_explanations.append(rule_metrics[rule_idx]["rule_description"])
                    
        row_data = raw_row.to_dict()
        if row_failed:
            feedback_payload = {
                "row_index": idx,
                "invalid_row_data": row_data,
                "regeneration_prompt": f"Failed: {violated_explanations}"
            }
            failed_rows_with_feedback.append(feedback_payload)
        else:
            passed_rows.append(row_data)
            
    # Convert aggregated metrics dictionary into a clean summary DataFrame
    coverage_df = pd.DataFrame(rule_metrics.values())
    
    return pd.DataFrame(passed_rows), failed_rows_with_feedback, coverage_df

if __name__ == "__main__":
    synthetic_file = "./data/diab_synthetic_output.csv"
    rules_file = "./output/automined_hard_rules.csv"
    
    try:
        clean_df, failed_queue, coverage_report = generate_rule_coverage_report(synthetic_file, rules_file)
        
        print("\n" + "═"*70)
        print(" GLOBAL CLINICAL RULE COVERAGE REPORT (EXPLAINABILITY CONSOLE)")
        print("═"*70)
        
        # Format and display the aggregated table clearly in the console
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 1000)
        
        # Only show columns useful for an auditor review
        print(coverage_report[['rule_id', 'rule_description', 'times_triggered', 'times_passed', 'times_failed']].to_string(index=False))
        print("═"*70)
        
        # Save summary report out for compliance logging
        coverage_report.to_csv("./data/clinical_rule_coverage_summary.csv", index=False)
        print(f"Summary report compiled and saved to './data/clinical_rule_coverage_summary.csv'")
        
    except FileNotFoundError:
        print(f"Error: Missing validation targets. Ensure '{synthetic_file}' and '{rules_file}' exist.")