# Fit calculation
## Mainland
### Read data and interpolate
``` Python
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
base_dir = "/Users/gurumakaza/Documents/data/IO table/cleaned_csv"
years_known = [2002, 2007, 2012, 2017]
years_all = list(range(2002, 2023))  # if you need for future steps
globals_dict = globals()

# --- Step 1: Load & Clean ---
for file in os.listdir(base_dir):
    if not file.endswith(".csv"):
        continue

    # Match file only if it includes one of the known years
    if any(f"_{year}_" in file for year in years_known):
        try:
            # Extract province and year from filename like 'jilin_2002_df.csv'
            parts = file.split("_")
            if len(parts) < 3:
                continue
            province = parts[0].lower()
            year = int(parts[1])
            var_name = f"{province}_{year}_df"

            # Read file
            file_path = os.path.join(base_dir, file)
            df = pd.read_csv(file_path, header=None,
                             low_memory=False)

            globals_dict[var_name] = df
            print(f"✓ Loaded: {var_name} → shape {df.shape}")


        except Exception as e:
            print(f"Failed to load {file}: {e}")















provinces = set(k.split('_')[0] for k in globals() if k.endswith("_2002_df"))




for province in provinces:
    # Check if all known years are present
    try:
        base_df = globals()[f"{province}_2017_df"]
        full_shape = base_df.shape
        interpolated = {}

        # Build time series per cell in bottom-right (5:, 5:)
        for i in range(5, full_shape[0]):
            for j in range(5, full_shape[1]):
                ts = {}
                for y in years_known:
                    df_y = globals().get(f"{province}_{y}_df")
                    if df_y is not None:
                        val = pd.to_numeric(df_y.iat[i, j], errors='coerce')
                        ts[y] = val if pd.notna(val) else np.nan
                s = pd.Series(ts).reindex(years_all)
                s_interp = s.interpolate(method='linear', limit_direction='both')
                interpolated[(i, j)] = s_interp

        # Rebuild annual DataFrames
        for y in years_all:
            df_new = base_df.copy()
            for i in range(5, full_shape[0]):
                for j in range(5, full_shape[1]):
                    df_new.iat[i, j] = interpolated[(i, j)].get(y, np.nan)
            globals()[f"{province}_{y}_df"] = df_new
            print(f"✓ Interpolated: {province}_{y}_df")

    except Exception as e:
        print(f"Interpolation failed for {province}: {e}")









output_dir = "/Users/gurumakaza/Documents/data/IO table/cleaned_csv"
os.makedirs(output_dir, exist_ok=True)

for var_name in globals():
    if var_name.endswith("_df") and isinstance(globals()[var_name], pd.DataFrame):
        df = globals()[var_name]
        filename = f"{var_name}.csv"
        filepath = os.path.join(output_dir, filename)
        try:
            df.to_csv(filepath, index=False, header=False)
            print(f"✓ Saved: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")











```

### Matrix correlation
```Python
import pandas as pd
import os
from glob import glob
from tqdm import tqdm

# Set your data folder
data_folder = '/Users/gurumakaza/Documents/data/IO table/cleaned_csv'

# Find all *_df.csv files
csv_files = sorted(glob(os.path.join(data_folder, '*_df.csv')))

# Final result will be collected here
all_qap_df = []






for file_path in tqdm(csv_files, desc="Processing files", unit="file"):
    

    # Extract year from filename (e.g. 'fujian_2004_df.csv' → 2004)
    basename = os.path.basename(file_path)
    province = basename.split("_")[0].capitalize()
    
    
    try:
        year = int([x for x in basename.split('_') if x.isdigit()][0])
    except:
        print(f"⚠️ Skipping file due to invalid year: {basename}")
        continue

    # ---- Insert all your code here ----
    # (start from reading the CSV down to computing `qap_df`)
    
    
    # Load the raw CSV without header interpretation
    df = pd.read_csv(file_path, header=None,
                     low_memory=False)
    



    # Get total number of rows and columns
    n_rows, n_cols = df.shape

    # Determine the start of data area
    start_row, start_col = 4, 4

    # Compute number of full 42x42 blocks
    block_size = 42

    valid_data_rows = (n_rows - start_row) // block_size * block_size
    valid_data_cols = (n_cols - start_col) // block_size * block_size

    # Compute the final row/column to keep
    end_row = start_row + valid_data_rows + 1
    end_col = start_col + valid_data_cols + 1

    # Trim the DataFrame: keep only valid area
    df_cleaned = df.iloc[:end_row, :end_col]

    df_cleaned = df_cleaned.fillna(0)

    # Define sector groups
    GROUPS = {
        'ESP': [7, 8, 19, 10, 20, 21],
        'EF': [22, 12, 14, 18, 16, 11],
        'WI': [25, 37, 26, 17, 23, 27],
        'ES': [5, 15, 13, 2, 3, 4],
        'RDS': [1, 9, 6, 30, 31, 24],
        'QoL': [38, 41, 28, 39, 40, 42],
        'LoD': [32, 33, 29, 34, 35, 36]
    }

    # Define required links
    LINKS = [
        ('ESP', 'QoL'),
        ('QoL', 'LoD'),
        ('LoD', 'RDS'),
        ('QoL', 'RDS'),
        ('EF', 'RDS'),
        ('RDS', 'WI'),
        ('EF', 'ES'),
        ('WI', 'ES'),
        ('ES', 'ESP'),
    ]

    def extract_group_links(df: pd.DataFrame, city: str, inv_type: str):
        """
        Extract group-to-group IO link matrices from a DataFrame with hierarchical headers.

        Args:
            df: Original DataFrame read from CSV.
            city: Target city name.
            inv_type: Investment type (内资, 港澳台, 外资).

        Returns:
            Dict of link matrices keyed by group names (e.g., "ESP_QoL").
        """
        # Extract actual matrix area
        df_data = df.iloc[4:, 4:]

        # Rebuild multi-level row/column indices
        df_data.columns = pd.MultiIndex.from_arrays([
            df.iloc[0, 4:].values,  # city row
            df.iloc[1, 4:].values,  # type row
            df.iloc[2, 4:].astype(int).values  # sector row
        ])
        df_data.index = pd.MultiIndex.from_arrays([
            df.iloc[4:, 0].values,  # city column
            df.iloc[4:, 1].values,  # type column
            df.iloc[4:, 2].astype(int).values  # sector column
        ])

        # Filter to city and type
        submatrix = df_data.loc[(city, inv_type), (city, inv_type)]

        # Reindex to sector numbers for clarity
        submatrix.index = submatrix.index.get_level_values(2)
        submatrix.columns = submatrix.columns.get_level_values(2)
        submatrix = submatrix.astype(float)

        # Extract group-to-group matrices
        result = {}
        for src, tgt in LINKS:
            try:
                result[f"{src}_{tgt}"] = submatrix.loc[GROUPS[src], GROUPS[tgt]]
            except KeyError:
                result[f"{src}_{tgt}"] = None  # optional: handle missing sectors

        return result










    # STEP 1: Define investment types
    investment_types = ['内资', '港澳台', '外资']

    # STEP 2: Extract all unique city names (from earlier)
    row_cities = df_cleaned.iloc[4:, 0].dropna().astype(str).unique()
    col_cities = df_cleaned.iloc[0, 4:].dropna().astype(str).unique()
    all_cities = pd.unique(pd.Series(list(row_cities) + list(col_cities)))
    all_cities = [c.strip() for c in all_cities if c.strip() != ""]
    all_cities.sort()
    # Clean up
    all_cities = [
        c.strip() for c in all_cities
        if isinstance(c, str) and c.strip() != "" and c.strip() != "0"
    ]






    def extract_city_type_matrix(df_cleaned: pd.DataFrame, city: str, inv_type: str) -> pd.DataFrame:
        # Find columns where row 0 = city and row 1 = type
        col_mask = (df_cleaned.iloc[0, 4:] == city) & (df_cleaned.iloc[1, 4:] == inv_type)
        col_indices = df_cleaned.columns[4:][col_mask]

        # Find rows where col 0 = city and col 1 = type
        row_mask = (df_cleaned.iloc[4:, 0] == city) & (df_cleaned.iloc[4:, 1] == inv_type)
        row_indices = df_cleaned.index[4:][row_mask]

        # Extract sector numbers (from row 2 and column 2)
        sector_cols = df_cleaned.loc[2, col_indices].astype(int).values
        sector_rows = df_cleaned.loc[row_indices, 2].astype(int).values

        # Extract matrix values
        matrix = df_cleaned.loc[row_indices, col_indices]

        # Set proper labels
        matrix.index = sector_rows
        matrix.columns = sector_cols

        return matrix.astype(float)









    # STEP 4: Loop over all (city, type) pairs
    city_type_matrices = {}

    for city in all_cities:
        city_type_matrices[city] = {}
        for inv_type in investment_types:
            matrix = extract_city_type_matrix(df_cleaned, city, inv_type)
            if matrix is not None:
                city_type_matrices[city][inv_type] = matrix
                
            else:
                print(f"❌ Missing: {city} - {inv_type}")
                
                


    from itertools import product

    LINKS = list(product(GROUPS.keys(), repeat=2))




    def extract_link_matrices(full_matrix):
        result = {}
        for src, tgt in LINKS:
            try:
                result[f"{src}_{tgt}"] = full_matrix.loc[GROUPS[src], GROUPS[tgt]]
            except KeyError:
                result[f"{src}_{tgt}"] = None  # if some sectors are missing
        return result





    all_links = {}  # all_links[city][type][link_name] = DataFrame

    for city, type_dict in city_type_matrices.items():
        all_links[city] = {}
        for inv_type, full_matrix in type_dict.items():
            all_links[city][inv_type] = extract_link_matrices(full_matrix)





    import numpy as np

    def qap_correlation(A, B):
        """
        Computes the QAP correlation between two matrices A and B.
        Equivalent to the Pearson correlation between the flattened off-diagonal elements.
        """
        A = np.array(A)
        B = np.array(B)

        # Mask out diagonal
        mask = ~np.eye(A.shape[0], dtype=bool)

        A_flat = A[mask].flatten()
        B_flat = B[mask].flatten()

        if np.std(A_flat) == 0 or np.std(B_flat) == 0:
            return np.nan  # Avoid divide-by-zero

        return np.corrcoef(A_flat, B_flat)[0, 1]





    

    records = []

    for city in all_links:
        for inv_type in all_links[city]:
            link_dict = all_links[city][inv_type]
            row = {'year': year, 'city': f"{province}_{city}", 'type': inv_type}
            used_pairs = set()

            for link_name, mat in link_dict.items():
                if mat is None:
                    row[link_name] = np.nan
                    continue

                src, tgt = link_name.split("_")
                reverse_name = f"{tgt}_{src}"

                # Avoid duplicate calculation
                if (tgt, src) in used_pairs:
                    continue

                mat_rev = link_dict.get(reverse_name)
                if mat_rev is not None:
                    corr = qap_correlation(mat, mat_rev.T)  # ✅ Transpose the reverse
                else:
                    corr = np.nan

                row[link_name] = corr
                used_pairs.add((src, tgt))

            records.append(row)

    qap_df = pd.DataFrame(records)

    # Add result to master list
    all_qap_df.append(qap_df)















# Combine all into one DataFrame
final_qap_df = pd.concat(all_qap_df, ignore_index=True)



# Make sure it's sorted
final_qap_df = final_qap_df.sort_values(['city', 'type', 'year'])

# Identify QAP columns (everything except metadata)
qap_cols = [col for col in final_qap_df.columns if col not in ['year', 'city', 'type']]

# Interpolate missing values within each city-type group
final_qap_df[qap_cols] = (
    final_qap_df
    .groupby(['city', 'type'])[qap_cols]
    .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
    .reset_index(drop=True)
)




# Optional: save to CSV
final_qap_df.to_csv("/Users/gurumakaza/Documents/data/full_qap_results.csv", index=False)
print("✅ Saved full results to full_qap_results.csv")


# Step 1: Load the CSV
file_path = "/Users/gurumakaza/Documents/data/full_qap_results.csv"
df = pd.read_csv(file_path)

# Step 2: Ensure correct types
df['year'] = df['year'].astype(int)
df['city'] = df['city'].astype(str)
df['type'] = df['type'].astype(str)

# Step 3: Identify QAP columns (exclude metadata)
qap_cols = [col for col in df.columns if col not in ['year', 'city', 'type']]

# Step 4: Sort and interpolate group-by-group
df = df.sort_values(['city', 'type', 'year'])

df[qap_cols] = (
    df.groupby(['city', 'type'])[qap_cols]
    .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
    .reset_index(drop=True)
)

# Step 5: Save cleaned version
cleaned_path = "/Users/gurumakaza/Documents/data/Correlation mainland.csv"
df.to_csv(cleaned_path, index=False)
print(f"✅ Cleaned and saved to {cleaned_path}")








```


### Summarize cells

```Python
import os
import pandas as pd
from tqdm import tqdm
from glob import glob

# Path to your folder
csv_dir = '/Users/gurumakaza/Documents/data/IO table/cleaned_csv'
csv_files = sorted(glob(os.path.join(csv_dir, '*_df.csv')))

# Constants
start_row, start_col = 4, 4
block_size = 42

total_cells = 0

for file_path in tqdm(csv_files, desc="Counting cells", unit="file"):
    df = pd.read_csv(file_path, header=None,
                     low_memory=False)
    n_rows, n_cols = df.shape

    valid_rows = (n_rows - start_row) // block_size * block_size
    valid_cols = (n_cols - start_col) // block_size * block_size

    total_cells += valid_rows * valid_cols

print(f"\n✅ Total cells across all valid blocks: {total_cells:,}")

```




## Hong Kong-Macau-Taiwan

### Read data and interpolate
```Python
import os
import pandas as pd
import numpy as np

# --- HMT Configuration ---
hmt_base_dirs = {
    'hkg': "/Users/gurumakaza/Documents/data/Hongkong Macao & Taiwan/Hongkong/Table",
    'mac': "/Users/gurumakaza/Documents/data/Hongkong Macao & Taiwan/Macau/Table",
    'twn': "/Users/gurumakaza/Documents/data/Hongkong Macao & Taiwan/Taiwan/Table"
}
years_known = [2002, 2007, 2012, 2017]
years_all = list(range(2002, 2023))
globals_dict = globals()

# --- Load and log1p HMT CSVs ---
for code, dir_path in hmt_base_dirs.items():
    for year in years_known:
        filename = f"{code.upper()}_{year}_sec42_iots_noncomp.csv"
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            print(f"❌ Missing: {file_path}")
            continue

        try:
            df_raw = pd.read_csv(file_path, header=None)
            df_42x42 = df_raw.iloc[1:43, 1:43].apply(pd.to_numeric, errors='coerce')

            # Apply log1p only to valid (non-null, non-negative) values
            df_logged = df_42x42.copy()
            df_logged = np.log1p(df_logged)

            var_name = f"{code}_{year}_df"
            globals_dict[var_name] = df_logged
            print(f"✓ Loaded + log1p: {var_name} → shape {df_logged.shape}")

        except Exception as e:
            print(f"Failed to load {file_path}: {e}")




# --- Interpolate/Extrapolate HMT ---
hmt_codes = ['hkg', 'mac', 'twn']

for code in hmt_codes:
    try:
        base_df = globals()[f"{code}_2017_df"]
        interpolated = {}

        for i in range(42):
            for j in range(42):
                ts = {}
                for y in years_known:
                    df_y = globals().get(f"{code}_{y}_df")
                    if df_y is not None:
                        val = pd.to_numeric(df_y.iat[i, j], errors='coerce')
                        ts[y] = val if pd.notna(val) else np.nan
                s = pd.Series(ts).reindex(years_all)
                s_interp = s.interpolate(method='linear', limit_direction='both')
                interpolated[(i, j)] = s_interp

        for y in years_all:
            df_new = base_df.copy()
            for i in range(42):
                for j in range(42):
                    df_new.iat[i, j] = interpolated[(i, j)].get(y, np.nan)
            globals()[f"{code}_{y}_df"] = df_new
            print(f"✓ Interpolated HMT: {code}_{y}_df")

    except Exception as e:
        print(f"Interpolation failed for {code}: {e}")









output_dir = "/Users/gurumakaza/Documents/data/IO table/cleaned_csv"
os.makedirs(output_dir, exist_ok=True)

for var_name in list(globals()):
    if var_name.endswith("_df") and isinstance(globals()[var_name], pd.DataFrame):
        df = globals()[var_name]
        filename = f"{var_name}.csv"
        filepath = os.path.join(output_dir, filename)
        try:
            df.to_csv(filepath, index=False, header=False)
            print(f"✓ Saved: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")







```


### Matrix correlation
```Python
from itertools import product
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm





from glob import glob

# HMT codes (lowercase)
hmt_codes = ['hkg', 'mac', 'twn']

# Base cleaned CSV directory
data_folder = '/Users/gurumakaza/Documents/data/IO table/cleaned_csv/HMT'

# Find only HMT files
csv_files = sorted([
    f for f in glob(os.path.join(data_folder, '*_df.csv'))
    if os.path.basename(f).split("_")[0].lower() in hmt_codes
])







# Define your groupings
GROUPS = {
    'ESP': [7, 8, 19, 10, 20, 21],
    'EF': [22, 12, 14, 18, 16, 11],
    'WI': [25, 37, 26, 17, 23, 27],
    'ES': [5, 15, 13, 2, 3, 4],
    'RDS': [1, 9, 6, 30, 31, 24],
    'QoL': [38, 41, 28, 39, 40, 42],
    'LoD': [32, 33, 29, 34, 35, 36]
}
LINKS = list(product(GROUPS.keys(), repeat=2))

def qap_correlation(A, B):
    A, B = np.array(A), np.array(B)
    mask = ~np.eye(A.shape[0], dtype=bool)
    A_flat, B_flat = A[mask], B[mask]
    if np.std(A_flat) == 0 or np.std(B_flat) == 0:
        return np.nan
    return np.corrcoef(A_flat, B_flat)[0, 1]

def extract_link_matrices(matrix):
    result = {}
    for src, tgt in LINKS:
        try:
            result[f"{src}_{tgt}"] = matrix.loc[GROUPS[src], GROUPS[tgt]]
        except KeyError:
            result[f"{src}_{tgt}"] = None
    return result

# HMT province codes
hmt_codes = ['hkg', 'mac', 'twn']

# Final results
all_qap_df = []

for file_path in tqdm(csv_files, desc="Processing files", unit="file"):
    basename = os.path.basename(file_path)
    parts = basename.split("_")
    province = parts[0].lower()

    try:
        year = int([x for x in parts if x.isdigit()][0])
    except:
        print(f"⚠️ Skipping file due to invalid year: {basename}")
        continue

    df = pd.read_csv(file_path, header=None)

    # --- HMT files ---
    if province in hmt_codes:
        try:
            mat = df.iloc[:42, :42].astype(float)
            mat.index = mat.columns = list(range(1, 43))  # sectors 1-42

            link_dict = extract_link_matrices(mat)
            row = {'year': year, 'city': province.upper(), 'type': 'HMT'}
            
            
            used_pairs = set()
            for link_name, submat in link_dict.items():
                if submat is None:
                    row[link_name] = np.nan
                    continue

                src, tgt = link_name.split("_")
                reverse_name = f"{tgt}_{src}"

                if (tgt, src) in used_pairs:
                    continue

                reverse_mat = link_dict.get(reverse_name)
                if reverse_mat is not None:
                    row[link_name] = qap_correlation(submat, reverse_mat.T)
                else:
                    row[link_name] = np.nan

                used_pairs.add((src, tgt))

            
            
            all_qap_df.append(pd.DataFrame([row]))
        except Exception as e:
            print(f"❌ Failed to process HMT file {basename}: {e}")
        continue

    # --- Mainland files (unchanged logic) ---
    try:
        n_rows, n_cols = df.shape
        start_row, start_col = 4, 4
        block_size = 42
        valid_data_rows = (n_rows - start_row) // block_size * block_size
        valid_data_cols = (n_cols - start_col) // block_size * block_size
        end_row = start_row + valid_data_rows + 1
        end_col = start_col + valid_data_cols + 1
        df_cleaned = df.iloc[:end_row, :end_col].fillna(0)

        # Extract all unique cities
        row_cities = df_cleaned.iloc[4:, 0].dropna().astype(str).unique()
        col_cities = df_cleaned.iloc[0, 4:].dropna().astype(str).unique()
        all_cities = pd.unique(pd.Series(list(row_cities) + list(col_cities)))
        all_cities = [c.strip() for c in all_cities if c.strip() and c.strip() != "0"]

        investment_types = ['内资', '港澳台', '外资']

        def extract_city_type_matrix(df_cleaned, city, inv_type):
            col_mask = (df_cleaned.iloc[0, 4:] == city) & (df_cleaned.iloc[1, 4:] == inv_type)
            col_indices = df_cleaned.columns[4:][col_mask]
            row_mask = (df_cleaned.iloc[4:, 0] == city) & (df_cleaned.iloc[4:, 1] == inv_type)
            row_indices = df_cleaned.index[4:][row_mask]
            if col_indices.empty or row_indices.empty:
                return None
            sector_cols = df_cleaned.loc[2, col_indices].astype(int).values
            sector_rows = df_cleaned.loc[row_indices, 2].astype(int).values
            matrix = df_cleaned.loc[row_indices, col_indices]
            matrix.index = sector_rows
            matrix.columns = sector_cols
            return matrix.astype(float)

        for city in all_cities:
            for inv_type in investment_types:
                mat = extract_city_type_matrix(df_cleaned, city, inv_type)
                if mat is None:
                    continue
                link_dict = extract_link_matrices(mat)
                row = {'year': year, 'city': f"{province.capitalize()}_{city}", 'type': inv_type}
                for link_name, submat in link_dict.items():
                    row[link_name] = qap_correlation(submat, submat.T) if submat is not None else np.nan
                all_qap_df.append(pd.DataFrame([row]))

    except Exception as e:
        print(f"❌ Failed to process mainland file {basename}: {e}")






# Combine all into one DataFrame
final_qap_df = pd.concat(all_qap_df, ignore_index=True)
final_qap_df = final_qap_df.sort_values(['city', 'type', 'year'])

qap_cols = [col for col in final_qap_df.columns if col not in ['year', 'city', 'type']]
final_qap_df[qap_cols] = (
    final_qap_df
    .groupby(['city', 'type'])[qap_cols]
    .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
    .reset_index(drop=True)
)

# Save results
final_qap_df.to_csv("/Users/gurumakaza/Documents/data/Correlation HMT.csv", index=False)
print("✅ Saved full results to full_qap_results.csv")

















```


# Regressions

```Python
import delimited "/Users/gurumakaza/Documents/data/Correlation mainland.csv", clear
encode city, gen(city_id)
encode type, gen(type_id)
keep if type == "内资"
duplicates drop city_id year, force
xtset city_id year
local depvars esp_qol qol_lod rds_lod qol_rds ef_rds rds_wi ef_es wi_es es_esp

cap erase qap_panel_fe.doc
cap erase qap_panel_fe.txt
foreach v of local depvars {
    xtreg `v' i.year, fe vce(robust)
    outreg2 using qap_panel_fe.doc, append ctitle(`v') alpha(0.001, 0.01, 0.05) bdec(3) tdec(3) addstat(R-squared, `e(r2)', N, e(N))
}
shellout using `"qap_panel_fe.doc"'
seeout using "qap_panel_fe.txt"
shellout using `"qap_panel_fe.doc"'
```
