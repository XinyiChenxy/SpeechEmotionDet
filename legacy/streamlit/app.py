"""Run: bash run_app.sh (or streamlit run app.py in the emotion2vec environment)."""
import hashlib
import json
import subprocess

import altair as alt
import pandas as pd
import streamlit as st

from ser.app_service import (BEST_CHECKPOINT, DISPLAY_LABELS,
                             BestModelPredictor, best_checkpoint_info)
from ser.common import LABELS, ROOT

st.set_page_config(page_title='声音里的情绪 · Speech Emotion Lab', page_icon='🎧', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1120px; padding-top: 2.6rem;}
h1 {letter-spacing: -.035em; font-weight: 750;}
[data-testid="stSidebar"] {background: #eef3f2;}
.eyebrow {color: #16776a; font-size: .76rem; letter-spacing: .16em; font-weight: 700; margin-bottom: .5rem;}
.intro {color: #56636c; margin-bottom: 1.5rem; line-height: 1.7;}
.empty-result {min-height: 235px; display:flex; align-items:center; justify-content:center;
 flex-direction:column; color:#6b7780; border:1px dashed #bfcfcb; border-radius:12px; background:#f7faf9;}
.empty-result strong {color:#344b46; margin-bottom:10px; font-size:1.12rem;}
div.stButton > button {min-height: 44px;}
</style>''', unsafe_allow_html=True)


def artifact_signature():
    # Invalidate Streamlit's resource cache if the best head or encoder changes.
    paths = [BEST_CHECKPOINT, BEST_CHECKPOINT.parent / 'history.jsonl',
             ROOT / 'checkpoints/emotion2vec_base/emotion2vec_base.pt',
             ROOT / 'checkpoints/emotion2vec_base/config.yaml']
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


@st.cache_data(show_spinner=False)
def model_info(signature):
    return best_checkpoint_info()


@st.cache_resource(show_spinner=False)
def predictor(signature):
    return BestModelPredictor()


def recognize(content, filename, signature):
    try:
        with st.spinner('正在分析声音，首次识别需要加载模型…'):
            result = predictor(signature).predict_bytes(content, filename)
        st.session_state['prediction'] = result
    except (ValueError, OSError, RuntimeError, FloatingPointError, subprocess.SubprocessError) as error:
        st.session_state.pop('prediction', None)
        st.error(f'无法识别这段音频：{error}')


st.markdown('<div class="eyebrow">SPEECH EMOTION LAB</div>', unsafe_allow_html=True)
st.title('声音里的情绪')
st.markdown('<p class="intro">上传一段语音，听一听，再看看声音中呈现的情绪。</p>', unsafe_allow_html=True)

try:
    signature = artifact_signature()
    info = model_info(signature)
except (OSError, ValueError, RuntimeError) as error:
    st.error(f'最佳模型尚未就绪：{error}')
    st.stop()

with st.sidebar:
    st.markdown('### 当前模型')
    st.success('验证集最佳模型 · 已保存')
    st.metric('最佳轮次', f"Epoch {info['best_epoch']}")
    st.metric('验证集 Macro-F1', f"{info['val_macro_f1']:.2%}")
    st.caption('依据验证集 Macro-F1 选择，early stopping 后加载最佳模型。')
    st.divider()
    st.write('**识别范围**')
    st.caption('愤怒 · 厌恶 · 恐惧 · 高兴 · 中性 · 悲伤')
    st.write('**使用方式**')
    st.caption('WAV / MP3，20 MB 以内、最长 30 秒。上传后点击“开始识别”。')
    with st.expander('模型详情'):
        st.write('emotion2vec_base · 冻结编码器')
        st.write(f"分类头：{info['head'].upper()}")
        st.code('outputs/cremad_emotion2vec_linear/best.pt', language=None)

left, right = st.columns([1, 1.08], gap='large')
content, filename = None, None
with left:
    st.subheader('01  输入音频')
    source = st.radio('选择音频来源', ['上传文件', '数据集示例'], horizontal=True)
    if source == '上传文件':
        uploaded = st.file_uploader('上传 WAV 或 MP3', type=['wav', 'mp3'], accept_multiple_files=False)
        if uploaded is not None:
            content, filename = uploaded.getvalue(), uploaded.name
    else:
        # Local demos make the interface immediately usable; predictions still
        # come exclusively from the saved best model, never from filenames.
        examples = {
            '示例 1 · 短句语音': ROOT / 'data/AudioWAV/1001_DFA_ANG_XX.wav',
            '示例 2 · 短句语音': ROOT / 'data/AudioWAV/1001_DFA_NEU_XX.wav',
            '示例 3 · 短句语音': ROOT / 'data/AudioWAV/1001_DFA_SAD_XX.wav',
        }
        selected = st.selectbox('选择一段示例', list(examples))
        path = examples[selected]
        if path.is_file():
            content, filename = path.read_bytes(), path.name
        else:
            st.info('本地示例不存在，请切换到上传文件。')
    input_id = hashlib.sha256(content).hexdigest() if content else None
    identity = (input_id, filename, signature)
    if st.session_state.get('input_identity') != identity:
        st.session_state.pop('prediction', None)
        st.session_state['input_identity'] = identity
    if content:
        st.audio(content, format='audio/mpeg' if filename.lower().endswith('.mp3') else 'audio/wav')
        st.caption(f'{filename} · {len(content) / 1024:.0f} KB')
    else:
        st.caption('支持单声道或立体声，采样率会自动处理。')
    if st.button('开始识别', type='primary', use_container_width=True, disabled=not content):
        recognize(content, filename, signature)
    st.caption('音频在当前容器内处理，临时上传文件在识别后删除。')

with right:
    st.subheader('02  情绪预测')
    result = st.session_state.get('prediction')
    if result is None:
        st.markdown('<div class="empty-result"><strong>等待一段声音</strong><span>选择音频后，点击“开始识别”查看结果。</span></div>', unsafe_allow_html=True)
    else:
        top, probability = st.columns([1.2, 1])
        top.metric('预测情绪', result['emotion_zh'])
        probability.metric('预测概率', f"{result['probabilities'][result['prediction']]:.2%}")
        st.caption(f"{result['prediction']} · 音频 {result['duration_seconds']:.2f} 秒 · 使用 Epoch {result['best_epoch']} 最佳模型")
        table = pd.DataFrame([{'emotion': DISPLAY_LABELS[label], 'label': label, 'probability': result['probabilities'][label]}
                              for label in LABELS])
        chart = alt.Chart(table).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('probability:Q', title='模型概率', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
            y=alt.Y('emotion:N', title=None, sort='-x'),
            color=alt.condition(alt.datum.label == result['prediction'], alt.value('#16776a'), alt.value('#c8dcd6')),
            tooltip=[alt.Tooltip('emotion:N', title='情绪'), alt.Tooltip('probability:Q', title='概率', format='.2%')]
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
        with st.expander('查看六类概率'):
            st.dataframe(table[['emotion', 'label', 'probability']].rename(columns={'emotion': '情绪', 'label': '英文标签', 'probability': '概率'}),
                         column_config={'概率': st.column_config.NumberColumn(format='%.4f')}, hide_index=True, use_container_width=True)
        st.download_button('下载识别结果 JSON', json.dumps(result, ensure_ascii=False, indent=2),
                           file_name='emotion_prediction.json', mime='application/json', use_container_width=True)

st.divider()
st.caption('CREMA-D 六类语音情绪识别课程项目 · emotion2vec 特征迁移学习')
