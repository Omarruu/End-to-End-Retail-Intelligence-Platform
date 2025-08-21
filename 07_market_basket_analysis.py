import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import time

print("⏳ Loading data...")
df = pd.read_parquet("data/cleaned_online_retail.parquet")

# 1) Filter out postage
print("🔍 Filtering out postage items...")
df = df[df.Description.str.upper() != "DOTCOM POSTAGE"]

# 2) Build baskets
print("🛒 Building baskets by invoice...")
start = time.time()
baskets = df.groupby("InvoiceNo")["Description"].apply(list)
print(f"  • {len(baskets)} baskets created in {time.time() - start:.1f}s")

# 3) One-hot encode
print("⚙️  Encoding transactions (one-hot)...")
te = TransactionEncoder()
start = time.time()
te_ary = te.fit(baskets).transform(baskets)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)
print(f"  • Encoded {basket_df.shape[0]} rows × {basket_df.shape[1]} items in {time.time() - start:.1f}s")

# 4) Apriori
print("📈 Mining frequent itemsets (min_support=0.01)...")
start = time.time()
freq_items = apriori(basket_df, min_support=0.01, use_colnames=True)
print(f"  • Found {len(freq_items)} frequent itemsets in {time.time() - start:.1f}s")

# 5) Association rules
print("🔗 Generating association rules (min_confidence=0.4, lift>1.2)...")
start = time.time()
rules = association_rules(freq_items, metric="confidence", min_threshold=0.4)
rules = rules[rules["lift"] > 1.2].sort_values(["confidence","lift"], ascending=False)
print(f"  • Generated {len(rules)} rules in {time.time() - start:.1f}s")

# 6) Save
print("💾 Saving outputs...")
freq_items.to_csv("data/frequent_itemsets.csv", index=False)
rules.to_csv("data/association_rules.csv", index=False)
print("✅ Done. Results saved to data/frequent_itemsets.csv and data/association_rules.csv")
print(rules.head(10))
