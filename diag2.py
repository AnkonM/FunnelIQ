"""
Isolate exact Plotly errors on the model page and OS bar chart.
Run: python diag2.py
"""
import pandas as pd, numpy as np, traceback
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── Load & train ──────────────────────────────────────────────
df = pd.read_csv('online_shoppers_intention.csv')
df['Revenue'] = df['Revenue'].astype(int)

NUM_COLS = ['Administrative','Administrative_Duration','Informational','Informational_Duration',
            'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']
CAT_COLS = ['Month','VisitorType']
X = df[NUM_COLS + CAT_COLS]; y = df['Revenue']
num_pipe = Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())])
cat_pipe = Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                     ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'))])
pre  = ColumnTransformer([('num',num_pipe,NUM_COLS),('cat',cat_pipe,CAT_COLS)])
pipe = Pipeline([('pre',pre),('clf',LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced'))])
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe.fit(X_tr,y_tr)
y_pred=pipe.predict(X_te); y_prob=pipe.predict_proba(X_te)[:,1]; y_te_arr=np.array(y_te)

cm_arr = np.array(confusion_matrix(y_te_arr,y_pred).tolist())
fpr_l,tpr_l,_ = roc_curve(y_te_arr,y_prob)
roc_auc_val = float(auc(fpr_l,tpr_l))
fpr_list = fpr_l.tolist(); tpr_list = tpr_l.tolist()
prob_df = pd.DataFrame({'Probability':y_prob.tolist(),
                        'Outcome':['Converted' if v==1 else 'Not Converted' for v in y_te_arr]})
ohe_names = pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(CAT_COLS)
all_feats = NUM_COLS+list(ohe_names); coefs=pipe.named_steps['clf'].coef_[0]
feat_imp = (pd.DataFrame({'Feature':all_feats,'Coefficient':coefs})
              .sort_values('Coefficient',ascending=False).reset_index(drop=True))

def test(name, fn):
    try:
        fn()
        print(f"  ✅  {name}")
    except Exception:
        print(f"  ❌  {name}")
        traceback.print_exc()
    print()

print("=== MODEL PAGE CHARTS ===")

# A: Confusion matrix
def cm_chart():
    cm_labels = ['Not Converted','Converted']
    fig = px.imshow(cm_arr, text_auto=True, color_continuous_scale='Blues',
                    x=cm_labels, y=cm_labels,
                    labels=dict(x='Predicted',y='Actual'), zmin=0)
    fig.update_traces(textfont_size=18)
    fig.to_json()   # forces full render
test("Confusion Matrix px.imshow", cm_chart)

# B: ROC Curve
def roc_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_list, y=tpr_list, mode='lines',
                             name=f'AUC={roc_auc_val:.3f}',
                             line=dict(color='#0ea5e9',width=2.5),
                             fill='tozeroy',fillcolor='rgba(14,165,233,0.07)'))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',
                             line=dict(color='#64748b',dash='dash',width=1.5)))
    fig.to_json()
test("ROC Curve go.Figure", roc_chart)

# C: Histogram
def hist_chart():
    fig = px.histogram(prob_df, x='Probability', color='Outcome',
                       barmode='overlay', nbins=35, opacity=0.75,
                       color_discrete_map={'Converted':'#0ea5e9','Not Converted':'#64748b'},
                       category_orders={'Outcome':['Not Converted','Converted']},
                       labels={'Probability':'Predicted Probability','Outcome':''})
    fig.to_json()
test("Probability Histogram px.histogram", hist_chart)

# D: Feature importance bar
def fi_chart():
    n_pos = int(min(8,(feat_imp['Coefficient']>0).sum()))
    n_neg = int(min(7,(feat_imp['Coefficient']<0).sum()))
    fi_show = pd.concat([
        feat_imp[feat_imp['Coefficient']>0].head(n_pos),
        feat_imp[feat_imp['Coefficient']<0].tail(n_neg),
    ]).reset_index(drop=True)
    fi_show['Color'] = fi_show['Coefficient'].apply(lambda x:'#22c55e' if x>0 else '#ef4444')
    fig = go.Figure(go.Bar(
        y=fi_show['Feature'].tolist(),
        x=fi_show['Coefficient'].tolist(),
        orientation='h',
        marker_color=fi_show['Color'].tolist(),
        text=[f'{v:.3f}' for v in fi_show['Coefficient']],
        textposition='outside',
    ))
    fig.to_json()
test("Feature Importance go.Bar", fi_chart)

print("=== FUNNEL PAGE CHARTS ===")

# E: OS bar
def os_chart():
    os_conv = (df.groupby('OperatingSystems')['Revenue']
               .agg(conv='sum',total='count').reset_index())
    os_conv['Conv %'] = (os_conv['conv']/os_conv['total']*100).round(1)
    os_conv['OS'] = 'OS ' + os_conv['OperatingSystems'].astype(str)
    os_conv = os_conv[os_conv['total']>50].sort_values('Conv %',ascending=False).reset_index(drop=True)
    print(f"    OS data ({len(os_conv)} rows):\n{os_conv[['OS','Conv %','total']].to_string()}")
    fig = px.bar(os_conv, x='OS', y='Conv %', text='Conv %',
                 color='OS',
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title='Conv. Rate by Operating System',
                 labels={'Conv %':'Conv. Rate (%)'},)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.to_json()
test("OS Bar chart", os_chart)

# F: Bounce bracket
def bounce_chart():
    bounce_bins = pd.cut(df['BounceRates'],bins=[0,.02,.05,.1,.2,1.0],
                         labels=['0–2%','2–5%','5–10%','10–20%','>20%'])
    bb = df.groupby(bounce_bins,observed=True)['Revenue'].mean().reset_index()
    bb.columns = ['Bounce Bracket','Conv Rate']
    bb['Conv %'] = (bb['Conv Rate']*100).round(1)
    bb['Bounce Bracket'] = bb['Bounce Bracket'].astype(str)
    print(f"    Bounce data:\n{bb.to_string()}")
    fig = px.bar(bb,x='Bounce Bracket',y='Conv %',text='Conv %',
                 color='Conv %',color_continuous_scale='RdYlGn',
                 title='Conv. Rate by Bounce Rate Bracket',labels={'Conv %':'Conv. Rate (%)'})
    fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
    fig.to_json()
test("Bounce Bar chart", bounce_chart)
