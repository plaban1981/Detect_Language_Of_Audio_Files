import streamlit as st
from PIL import Image
import torch
import whisper
import joblib
import os
import json
import pydub
#
st.set_page_config(page_title="Detect the language of the audio file App", layout="centered")
st.image("image.jpg", caption='Detect the language of the audio file')
# page header
#st.title(f"English Text to Hindi Translation App")
with st.form("Prediction_form"):
   uploadFile = st.file_uploader(label = "Please upload your file ",type=["mp3"])
   if uploadFile is not None:
       audio = pydub.AudioSegment.from_mp3(uploadFile)
       file_type = 'mp3'
       #filename = uploadFile.name+".mp3"
       audio.export(uploadFile.name, format="mp3")
       audio1 = open(uploadFile.name, "rb")
       audio_bytes = audio1.read() #reading the file
       st.audio(audio_bytes, format='audio/ogg') #displaying the audio
   #
   submit = st.form_submit_button("Detect Language")
   #
   if submit:
        model = whisper.load_model("large").to('cpu')
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(uploadFile.name)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # detect the spoken language
        _, probs = model.detect_language(mel)    
        # decode the audio
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(model, mel, options)
        languages = {"af_za": "Afrikaans", "am_et": "Amharic", 
                   "ar_eg": "Arabic", "as_in": "Assamese", 
                   "az_az": "Azerbaijani", "be_by": "Belarusian", 
                   "bg_bg": "Bulgarian", "bn_in": "Bengali", 
                   "bs_ba": "Bosnian", "ca_es": "Catalan", 
                   "cmn_hans_cn": "Chinese", "cs_cz": "Czech", 
                   "cy_gb": "Welsh", "da_dk": "Danish", 
                   "de_de": "German", "el_gr": "Greek", 
                   "en_us": "English", "es_419": "Spanish", 
                   "et_ee": "Estonian", "fa_ir": "Persian", 
                   "fi_fi": "Finnish", "fil_ph": "Tagalog", 
                   "fr_fr": "French", "gl_es": "Galician", 
                   "gu_in": "Gujarati", "ha_ng": "Hausa", 
                   "he_il": "Hebrew", "hi_in": "Hindi", 
                   "hr_hr": "Croatian", "hu_hu": "Hungarian", 
                   "hy_am": "Armenian", "id_id": "Indonesian", 
                   "is_is": "Icelandic", "it_it": "Italian", 
                   "ja_jp": "Japanese", "jv_id": "Javanese", 
                   "ka_ge": "Georgian", "kk_kz": "Kazakh", 
                   "km_kh": "Khmer", "kn_in": "Kannada", 
                   "ko_kr": "Korean", "lb_lu": "Luxembourgish", 
                   "ln_cd": "Lingala", "lo_la": "Lao", 
                   "lt_lt": "Lithuanian", "lv_lv": "Latvian", 
                   "mi_nz": "Maori", "mk_mk": "Macedonian", 
                   "ml_in": "Malayalam", "mn_mn": "Mongolian", 
                   "mr_in": "Marathi", "ms_my": "Malay", 
                   "mt_mt": "Maltese", "my_mm": "Myanmar", 
                   "nb_no": "Norwegian", "ne_np": "Nepali", 
                   "nl_nl": "Dutch", "oc_fr": "Occitan", 
                   "pa_in": "Punjabi", "pl_pl": "Polish", 
                   "ps_af": "Pashto", "pt_br": "Portuguese", 
                   "ro_ro": "Romanian", "ru_ru": "Russian", 
                   "sd_in": "Sindhi", "sk_sk": "Slovak", 
                   "sl_si": "Slovenian", "sn_zw": "Shona", "so_so": "Somali", "sr_rs": "Serbian", "sv_se": "Swedish", "sw_ke": "Swahili", "ta_in": "Tamil", "te_in": "Telugu", "tg_tj": "Tajik", "th_th": "Thai", "tr_tr": "Turkish", "uk_ua": "Ukrainian", 
                   "ur_pk": "Urdu", "uz_uz": "Uzbek", "vi_vn": "Vietnamese", "yo_ng": "Yoruba"}
        decode_language = {k.split("_")[0]: v for k,v in languages.items()}
        language = decode_language[result.language]
        text = result.text
        out = f"Language Detected: {language} Text: {text}"
        # output header
        st.header("Summary Gnereated")
        # output results
        st.success(f"Translated Text : {out}")