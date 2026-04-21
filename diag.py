import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv('online_shoppers_intention.csv')
df['Revenue'] = df['Revenue'].astype(int)

def chk(name, fn):
    try:
        r = fn()
        print(f"OK  {name}: {r}")
    except Exception as e:
        print(f"ERR {name}: {e}")

# --- Overview charts ---
chk("visitor_type", lambda: df.groupby('VisitorType')['Revenue'].mean().to_dict())

month_order = ['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
def mo_test():
    mo = df.groupby('Month')['Revenue'].mean().reset_index()
    mo['Month'] = pd.Categorical(mo['Month'], categories=month_order, ordered=True)
    mo = mo.sort_values('Month').dropna()
    return f"rows={len(mo)} months={mo['Month'].tolist()}"
chk("monthly_trend", mo_test)

def tt_test():
    tt = df.groupby('TrafficType')['Revenue'].agg(count='count', conversions='sum').reset_index()
    tt['Conv Rate %'] = (tt['conversions'] / tt['count'] * 100).round(1)
    tt['TrafficType'] = 'Type ' + tt['TrafficType'].astype(str)
    tt['size_col'] = tt['conversions'].clip(lower=1)
    tt_top = tt.nlargest(10, 'count')
    return f"rows={len(tt_top)} min_size={tt_top['size_col'].min()}"
chk("traffic_type_scatter", tt_test)

def pv_test():
    sample = df.sample(1000, random_state=42).copy()
    sample['Outcome'] = sample['Revenue'].map({1: 'Converted', 0: 'Not Converted'})
    counts = sample['Outcome'].value_counts().to_dict()
    return f"counts={counts}"
chk("page_value_scatter", pv_test)

# --- Funnel charts ---
def wk_test():
    wk = df.groupby('Weekend')['Revenue'].mean().reset_index()
    wk['Label'] = wk['Weekend'].map({True:'Weekend', False:'Weekday'})
    wk['Rate %'] = (wk['Revenue'] * 100).round(1)
    return f"rows={len(wk)} vals={wk['Rate %'].tolist()}"
chk("weekend_pie", wk_test)

def bounce_test():
    bounce_bins = pd.cut(df['BounceRates'], bins=[0,.02,.05,.1,.2,1.0],
                         labels=['0-2%','2-5%','5-10%','10-20%','>20%'])
    bb = df.groupby(bounce_bins, observed=True)['Revenue'].mean().reset_index()
    bb['Rate %'] = (bb['Revenue'] * 100).round(1)
    return f"rows={len(bb)} vals={bb['Rate %'].tolist()}"
chk("bounce_bins", bounce_test)

def os_test():
    os_conv = df.groupby('OperatingSystems')['Revenue'].agg(['mean','count']).reset_index()
    os_conv.columns = ['OS','Conv Rate','Sessions']
    os_conv = os_conv[os_conv['Sessions'] > 50].copy()
    os_conv['OS Label'] = 'OS ' + os_conv['OS'].astype(str)
    os_conv['Conv Rate %'] = (os_conv['Conv Rate']*100).round(1)
    return f"rows={len(os_conv)} vals={os_conv['Conv Rate %'].tolist()}"
chk("os_bar", os_test)

# --- Model charts ---
NUM_COLS = ['Administrative','Administrative_Duration','Informational','Informational_Duration',
            'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']
CAT_COLS = ['Month','VisitorType']
X = df[NUM_COLS + CAT_COLS]; y = df['Revenue']
num_pipe = Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())])
cat_pipe = Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                     ('o',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'))])
pre = ColumnTransformer([('num',num_pipe,NUM_COLS),('cat',cat_pipe,CAT_COLS)])
pipe = Pipeline([('pre',pre),('clf',LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced'))])
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe.fit(X_tr,y_tr)
y_pred=pipe.predict(X_te); y_prob=pipe.predict_proba(X_te)[:,1]; y_te_arr=np.array(y_te)

def cm_test():
    cm = np.array(confusion_matrix(y_te_arr, y_pred).tolist())
    return f"shape={cm.shape} values={cm.tolist()}"
chk("confusion_matrix", cm_test)

def roc_test():
    fpr,tpr,_ = roc_curve(y_te_arr, y_prob)
    return f"auc={auc(fpr,tpr):.4f} fpr_len={len(fpr)}"
chk("roc_curve", roc_test)

def prob_test():
    prob_df = pd.DataFrame({'Probability': y_prob,
                            'Outcome': ['Converted' if v==1 else 'Not Converted' for v in y_te_arr]})
    return f"rows={len(prob_df)} counts={prob_df['Outcome'].value_counts().to_dict()}"
chk("prob_dist", prob_test)

def feat_test():
    ohe_names = pipe.named_steps['pre'].named_transformers_['cat'].named_steps['o'].get_feature_names_out(CAT_COLS)
    all_feats = NUM_COLS + list(ohe_names)
    coefs = pipe.named_steps['clf'].coef_[0]
    feat_imp = pd.DataFrame({'Feature':all_feats,'Coefficient':coefs}).sort_values('Coefficient',ascending=False).reset_index(drop=True)
    n_pos = min(8,(feat_imp['Coefficient']>0).sum())
    n_neg = min(7,(feat_imp['Coefficient']<0).sum())
    fi = pd.concat([feat_imp[feat_imp['Coefficient']>0].head(n_pos),
                    feat_imp[feat_imp['Coefficient']<0].tail(n_neg)]).reset_index(drop=True)
    fi['Color'] = fi['Coefficient'].apply(lambda x: '#22c55e' if x>0 else '#ef4444')
    return f"rows={len(fi)} top={fi['Feature'].iloc[0]}"
chk("feature_importance", feat_test)
