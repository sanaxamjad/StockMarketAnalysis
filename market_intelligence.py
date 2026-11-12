
import html
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title='Market Intelligence', page_icon='◈', layout='wide')
BG='#0B1018'; PANEL='#151D29'; EDGE='#293446'; TEXT='#E9F1F8'; MUTED='#A0AFC0'; GREEN='#51E3B0'; RED='#FF8592'; GOLD='#F2C777'
st.markdown(f'''<style>
:root{{color-scheme:dark}}html,body,.stApp{{background:{BG};color:{TEXT}}}
[data-testid="stSidebar"]{{background:#101824;border-right:1px solid {EDGE}}}
.block-container{{max-width:1450px;padding-top:1.8rem}}
h1,h2,h3,p,label,[data-testid="stMarkdownContainer"]{{color:{TEXT}}}
[data-testid="stMetric"],div[data-testid="stVerticalBlockBorderWrapper"]>div{{background:{PANEL};border-radius:16px}}
[data-testid="stMetric"]{{border:1px solid {EDGE};padding:18px}}
[data-testid="stMetricLabel"]{{color:{MUTED}}}
[data-testid="stMetricValue"]{{color:{TEXT}}}
.stButton button,.stDownloadButton button{{border-radius:12px;background:#1C293A;color:{TEXT};border:1px solid {EDGE}}}
.stButton button:hover,.stDownloadButton button:hover{{border-color:{GREEN};color:{GREEN}}}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{{background:{PANEL}!important;color:{TEXT}!important}}
[data-testid="stTabs"] button{{color:{MUTED}}}
[data-testid="stTabs"] button[aria-selected="true"]{{color:{GREEN}}}
.hero{{padding:26px 30px;border:1px solid {EDGE};border-radius:22px;background:linear-gradient(120deg,#17283A,#111923);margin-bottom:20px}}
.hero h1{{margin:0;color:{TEXT}}}.hero p{{color:{MUTED};margin:8px 0 0}}
.note{{background:{PANEL};border:1px solid {EDGE};border-radius:14px;padding:18px;margin:10px 0;color:{TEXT}}}
</style>''', unsafe_allow_html=True)

DB=Path(__file__).with_name('market_journal.db')
def db():
    conn=sqlite3.connect(DB)
    conn.execute('CREATE TABLE IF NOT EXISTS journal (id INTEGER PRIMARY KEY, date TEXT, ticker TEXT, stance TEXT, price REAL, thesis TEXT)')
    conn.commit()
    return conn

def fmt(v): return f'${v:,.2f}'
def pct(v): return f'{v:+.2f}%'
def chart_layout(fig,height=380):
    fig.update_layout(height=height,paper_bgcolor=PANEL,plot_bgcolor=PANEL,font=dict(color=MUTED),margin=dict(l=12,r=12,t=20,b=15),xaxis=dict(showgrid=False),yaxis=dict(gridcolor=EDGE),hovermode='x unified',legend=dict(orientation='h',y=1.13))
    return fig

@st.cache_data(ttl=300,show_spinner=False)
def prices(ticker,period='1y',interval='1d'):
    d=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if d.empty:return pd.DataFrame()
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=['Close']).copy()

@st.cache_data(ttl=900,show_spinner=False)
def headlines(ticker):
    url='https://news.google.com/rss/search?q='+quote_plus(f'{ticker} stock earnings company')+'&hl=en-US&gl=US&ceid=US:en'
    try:
        r=requests.get(url,timeout=10,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status()
        feed=feedparser.parse(r.content)
        return [{'title':e.get('title',''),'link':e.get('link',''),'published':e.get('published','')} for e in feed.entries[:8]]
    except requests.RequestException:return []

sentiment=SentimentIntensityAnalyzer()
def news_analysis(items):
    out=[]
    for a in items:
        s=sentiment.polarity_scores(a['title'])['compound']
        out.append({**a,'score':s,'tone':'Positive' if s>.1 else 'Negative' if s<-.1 else 'Mixed/neutral'})
    return out

def analyze(df):
    if len(df)<55:return None
    close=df['Close']; p=float(close.iloc[-1]); ma20=close.rolling(20).mean();ma50=close.rolling(50).mean()
    delta=close.diff();gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean();loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    if loss.iloc[-1]==0:rsi=100. if gain.iloc[-1]>0 else 50.
    else:rsi=float(100-100/(1+gain.iloc[-1]/loss.iloc[-1]))
    momentum=(p/float(close.iloc[-21])-1)*100
    positives=[];warnings=[]
    if p>ma20.iloc[-1]:positives.append('Price is above its recent average')
    else:warnings.append('Price is below its recent average')
    if p>ma50.iloc[-1]:positives.append('The longer price trend is positive')
    else:warnings.append('The longer price trend is weak')
    if momentum>0:positives.append('The stock gained over the last ~month')
    else:warnings.append('The stock declined over the last ~month')
    if 40<=rsi<=65:positives.append('Price momentum is moderate')
    elif rsi>70:warnings.append('A rapid rise may be overheated')
    elif rsi<30:warnings.append('Selling pressure has been heavy')
    score=len(positives)
    label='Research a possible entry' if score>=3 and rsi<70 else 'Review risk / thesis' if score<=1 else 'Watch and wait'
    returns=close.pct_change().dropna();vol=float(returns.tail(20).std()*np.sqrt(252)*100)
    tr=pd.concat([(df['High']-df['Low']), (df['High']-close.shift()).abs(),(df['Low']-close.shift()).abs()],axis=1).max(axis=1)
    atr=float(tr.rolling(14).mean().iloc[-1])
    return dict(price=p,score=score,label=label,positives=positives,warnings=warnings,momentum=momentum,rsi=rsi,vol=vol,atr=atr,ma20=ma20,ma50=ma50)

def price_chart(df,analysis,days=90):
    d=df.tail(days)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.index,y=d.Close,name='Share price',line=dict(color=GREEN,width=3),fill='tozeroy',fillcolor='rgba(81,227,176,.06)'))
    fig.add_trace(go.Scatter(x=d.index,y=analysis['ma20'].reindex(d.index),name='Recent average',line=dict(color='#92A5C2',dash='dot')))
    return chart_layout(fig,420)

st.sidebar.title('◈ Market Intelligence')
watch_text=st.sidebar.text_area('Your watchlist','NVDA, AAPL, MSFT, AMD, CRDO, VOO',height=100)
watch=list(dict.fromkeys(x.strip().upper() for x in watch_text.split(',') if x.strip()))[:15]
selected=st.sidebar.selectbox('Explore a stock',watch if watch else ['NVDA'])
custom=st.sidebar.text_input('Or enter a ticker').strip().upper()
ticker=custom or selected
period=st.sidebar.select_slider('Chart history',['1mo','3mo','6mo','1y','2y'],value='3mo')
if st.sidebar.button('↻ Refresh now',use_container_width=True):st.cache_data.clear();st.rerun()
st.sidebar.caption('Checks every 5 minutes while the app is open. Yahoo Finance data may be delayed. No orders are placed.')

st.markdown('<div class="hero"><h1>Market Intelligence</h1><p>Your personal market radar · simple explanations · news context · decision journal</p></div>',unsafe_allow_html=True)
tabs=st.tabs(['Overview','Opportunity radar','Compare stocks','What-if lab','Research journal'])

@st.fragment(run_every=300)
def overview():
    with st.spinner('Checking prices and headlines...'):
        df=prices(ticker,'2y');intraday=prices(ticker,'5d','5m');items=news_analysis(headlines(ticker))
    a=analyze(df)
    if a is None:st.error('Not enough price history for this stock.');return
    latest=float(intraday.Close.iloc[-1]) if not intraday.empty else a['price']
    quote_time=str(intraday.index[-1]) if not intraday.empty else str(df.index[-1])
    # Previous close is the last completed daily bar when intraday exists; during market close use previous completed day.
    prev=float(df.Close.iloc[-1]) if not intraday.empty else float(df.Close.iloc[-2])
    change=(latest/prev-1)*100 if prev else 0
    st.subheader(f'{ticker} · At a glance')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Latest available price',fmt(latest),pct(change))
    c2.metric('Past ~month',pct(a['momentum']))
    c3.metric('Positive signs',f"{a['score']} / 4")
    c4.metric('Price swings','High' if a['vol']>50 else 'Moderate' if a['vol']>25 else 'Lower')
    st.caption('Last price observation: '+quote_time+' · May be delayed; daily trend indicators update after daily bars close.')
    left,right=st.columns([1.8,1])
    with left:
        st.plotly_chart(price_chart(df,a,{'1mo':22,'3mo':66,'6mo':132,'1y':252,'2y':504}[period]),use_container_width=True)
        st.caption('Green = historical price · dotted = recent average. Past performance does not predict future returns.')
    with right:
        st.markdown('### What does it mean?')
        st.markdown(f'<div class="note"><strong>{html.escape(a["label"])}</strong><p style="color:{MUTED}">This is a technical research cue, not an instruction to trade. Consider news, valuation, and your time horizon.</p></div>',unsafe_allow_html=True)
        st.write('**Encouraging signs**')
        for item in a['positives']:st.write('✓ '+item)
        st.write('**Things to watch**')
        for item in a['warnings']:st.write('• '+item)
    st.divider()
    st.subheader('News radar')
    if items:
        positives=sum(x['tone']=='Positive' for x in items);negative=sum(x['tone']=='Negative' for x in items)
        st.caption(f'{positives} positive-sounding · {negative} negative-sounding · {len(items)-positives-negative} neutral headlines. Headline tone is not a forecast.')
        for item in items:
            with st.container(border=True):
                st.markdown('**'+html.escape(item['title'])+'**')
                st.caption(item['tone']+' · '+item['published'])
                if item['link'].startswith('https://'):st.link_button('Read original article',item['link'])
    else:st.info('No headlines available right now. News feeds can fail or be incomplete.')
    st.subheader('Price scenario explorer')
    change_pct=st.slider('Imagine the stock changes by...',-30,30,5,key='overview_scenario')
    st.metric('Hypothetical share price',fmt(latest*(1+change_pct/100)),pct(change_pct))
    st.caption('A scenario, not a prediction.')
    with st.expander('Explain every investing term'):
        st.write('**Momentum:** whether the price is higher or lower than about a month ago. **Recent average:** a smoothed price line. **Volatility:** how widely prices swing. **Positive signs:** a count of four simple trend checks, not a probability of profit. **Headline sentiment:** wording of titles, not the effect on a company.')
    st.caption('Dashboard check: '+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))

with tabs[0]:overview()

@st.fragment(run_every=300)
def radar():
    st.subheader('Opportunity radar')
    st.caption('Rank your watchlist by price-trend strength. A high score is not proof that a stock is undervalued.')
    rows=[]
    for sym in watch:
        try:
            d=prices(sym,'1y');a=analyze(d)
            if a:rows.append({'Stock':sym,'Price':a['price'],'Positive signs':a['score'],'1-month change %':round(a['momentum'],2),'Price swings %':round(a['vol'],1),'Research cue':a['label']})
        except Exception:pass
    if rows:
        table=pd.DataFrame(rows).sort_values(['Positive signs','1-month change %'],ascending=False)
        st.dataframe(table,hide_index=True,use_container_width=True,column_config={'Price':st.column_config.NumberColumn(format='$%.2f'),'1-month change %':st.column_config.NumberColumn(format='%.2f%%'),'Price swings %':st.column_config.NumberColumn(format='%.1f%%')})
        st.download_button('Download watchlist report',table.to_csv(index=False),'market_radar.csv','text/csv')
        fig=go.Figure(go.Bar(x=table['Stock'],y=table['1-month change %'],marker_color=[GREEN if v>=0 else RED for v in table['1-month change %']]))
        st.plotly_chart(chart_layout(fig,330),use_container_width=True)
    else:st.info('No watchlist data available.')
with tabs[1]:radar()

with tabs[2]:
    st.subheader('Head-to-head stock comparison')
    options=watch if len(watch)>=2 else ['NVDA','AAPL']
    c1,c2=st.columns(2)
    with c1:first=st.selectbox('First stock',options,index=0)
    with c2:second=st.selectbox('Second stock',options,index=1 if len(options)>1 else 0)
    if first==second:st.info('Choose two different stocks.')
    else:
        d1=prices(first,'1y');d2=prices(second,'1y')
        if not d1.empty and not d2.empty:
            merged=pd.concat([d1.Close.rename(first),d2.Close.rename(second)],axis=1).dropna().tail(126)
            if len(merged)>1:
                normalized=merged.div(merged.iloc[0]).mul(100)
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=normalized.index,y=normalized[first],name=first,line=dict(color=GREEN,width=3)))
                fig.add_trace(go.Scatter(x=normalized.index,y=normalized[second],name=second,line=dict(color='#A8A4FF',width=3)))
                st.plotly_chart(chart_layout(fig,420),use_container_width=True)
                st.caption('Both start at 100 so you can compare percentage performance, not share prices.')
        else:st.warning('Could not load both stocks.')

with tabs[3]:
    st.subheader('What-if investment lab')
    st.caption('Explore possible outcomes, not predictions or investment recommendations.')
    amount=st.number_input('Amount you might invest ($)',min_value=0.0,value=500.0,step=50.0)
    move=st.slider('What if the share price changes?',-80,150,20)
    result=amount*(1+move/100)
    x,y,z=st.columns(3)
    x.metric('Starting amount',fmt(amount))
    y.metric('Hypothetical ending value',fmt(result))
    z.metric('Gain or loss',fmt(result-amount),pct(move))
    st.caption('Excludes taxes, dividends, fees and slippage. A stock can lose most or all of its value.')

with tabs[4]:
    st.subheader('Investment research journal')
    st.caption('Record your thinking before a trade so you can revisit the reasoning later. Stored locally on this computer.')
    with st.form('journal_form'):
        jt=st.text_input('Stock',value=ticker).strip().upper()
        stance=st.selectbox('Your idea',['Research buying','Watch','Review holding','Research selling'])
        thesis=st.text_area('Why are you considering this? What would change your mind?')
        submitted=st.form_submit_button('Save research note')
        if submitted:
            if not jt or not thesis.strip():st.warning('Add a ticker and a short explanation.')
            else:
                d=prices(jt,'1y');price=float(d.Close.iloc[-1]) if not d.empty else None
                conn=db()
                conn.execute('INSERT INTO journal(date,ticker,stance,price,thesis) VALUES(?,?,?,?,?)',(datetime.now(timezone.utc).isoformat(),jt,stance,price,thesis.strip()))
                conn.commit();conn.close();st.success('Research note saved locally.')
    conn=db();notes=pd.read_sql_query('SELECT date,ticker,stance,price,thesis FROM journal ORDER BY id DESC LIMIT 100',conn);conn.close()
    if not notes.empty:
        st.dataframe(notes,use_container_width=True,hide_index=True)
        st.download_button('Export journal',notes.to_csv(index=False),'research_journal.csv','text/csv')
    else:st.info('No saved notes yet.')

st.divider()
st.caption('Educational research tool · 5-minute checks are not a real-time exchange feed · Yahoo Finance and Google News may be delayed or unavailable · Signals are not personalized financial advice.')

