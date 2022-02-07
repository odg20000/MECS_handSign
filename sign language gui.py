#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_mic.py
"""

from __future__ import division
# ----------------
import random

# ---------------
index_num = -1
# ---------------
# ----------------

import cv2
import threading

from PyQt5 import QtWidgets
from PyQt5 import QtGui

from PyKomoran import *

import argparse
from jamostoolkit import JamosSeparator
from hangul_utils import join_jamos

import cv2
import mediapipe as mp
import preprocessing as pre
from tensorflow import keras
import numpy as np
import joblib

# =====================================================================================================================================

# kms
alph_dic = {'ga': 'ㄱ', 'ba': 'ㅂ', 'woo': 'ㅜ', 'yeo': 'ㅕ', 'aa': 'ㅇ', 'da': 'ㄷ', 'ae': 'ㅐ', 'ra': 'ㄹ', 'eo': 'ㅓ',
            'sa': 'ㅅ', 'ui': 'ㅢ', 'yae': 'ㅒ', 'ya': 'ㅑ', 'ja': 'ㅈ', 'cha': 'ㅊ', 'ka': 'ㅋ', 'ta': 'ㅌ', 'ha': 'ㅎ',
            'a': 'ㅏ', 'oh': 'ㅗ', 'oe': 'ㅚ', 'yo': 'ㅛ', 'wi': 'ㅟ', 'eu': 'ㅡ', 'lee': 'ㅣ'}

sc = joblib.load('sc_hinge_v10.pkl')

hangul_result = []


def signal_detection(data):
    detected = 0
    present = signal_detection.signal_present = data
    previous = signal_detection.signal_previous
    signal_detection.signal_times

    if present == previous:
        signal_detection.signal_times += 1
        if signal_detection.signal_times >= signal_detection.signal_threshold:
            print('Signal Detected : ', present)
            detected = 1
            signal_detection.signal_times = 0
    else:
        signal_detection.signal_times = 0
    # print('DEBUG::: STATIC VAR CHECK ::: ', signal_detection.signal_times)
    signal_detection.signal_previous = present
    if detected == 1:
        return present
    else:
        return 0


signal_detection.signal_times = 0
signal_detection.signal_previous = ['*']
signal_detection.signal_present = ['*']
signal_detection.signal_threshold = 60


def axis_move(data):
    x_zero = data[0]
    y_zero = data[21]
    z_zero = data[42]
    for i in range(0, 21):
        data[i] = data[i] - x_zero
    for i in range(21, 42):
        data[i] = data[i] - y_zero
    for i in range(42, 63):
        data[i] = data[i] - z_zero
    return data


# 형태소 분석기

Noun = ['NNG', 'NNP', 'NNB', 'NNBC', 'NP']
JoSa = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC']
Predicate = ['VV', 'VA', 'VX', 'EP', 'EF', 'EC', 'VV+EC', 'VA+EC', 'VCN', 'VX+EC', 'VX+EP', 'XSV',
             'VCP']  # E로 시작하는건 어미 - 서술어에만 쓰이는지는 모르겠음
Predicate_plus = ['VV+EC', 'VA+EC', 'VX+EC', "VX+EP"]
time_adverbs = [['어제', '이전', '그제'], ['현재', '지금', '이번'], ['내일', '나중', '곧', '모레']]
Ep = [['었', '았었', '었었', '았', '였'], ['는', 'ㄴ'], ['겠', '것']]


def jamos_test(text):
    parser = argparse.ArgumentParser(description='Jamos toolkit')
    parser.add_argument('--string', default=text, help='Please enter the string you want to separator.')
    args = parser.parse_args()
    jamos = JamosSeparator(args.string)
    jamos.run()
    print(jamos.get())
    return jamos.get()


def find_N_Josa(pos, i):  ## 명사 뒤에 조사가 있으면 명사에 조사 표시넣기
    j = 1
    while 1:
        for x in JoSa:
            if pos[i + j][1] == x:
                pos[i].append(pos[i + j][1])
                # print(pos[i])
                return
        j = j + 1


# def mecab(text):
#    pos = mecab.pos(text)
#    return pos

def N_J_processing_in_pos(pos):
    # 명사 찾는 for문
    for i in range(len(pos)):
        for noun in Noun:
            try:
                if pos[i][1] == noun:
                    find_N_Josa(pos, i)
            except IndexError as E:
                pass

    # 조사 없애는 for 문
    for i in range(len(pos)):
        for josa in JoSa:
            try:
                if pos[i][1] == josa:
                    del pos[i]
            except IndexError as E:
                pass
    # print(pos)
    return pos


def P_processing_in_text(pos):
    for i in range(len(pos)):
        for predicate in Predicate:
            if pos[i][1] == predicate:
                pos[i].append('P')
    # print(pos)
    return pos


def time_adverb(pos):
    for i in range(len(pos)):
        if pos[i][1] == 'MAG':
            if pos[i][0] in time_adverbs[0]:
                pos[i].append('past')
            elif pos[i][0] in time_adverbs[1]:
                pos[i].append('present')
            elif pos[i][0] in time_adverbs[2]:
                pos[i].append('future')
    # print(pos)
    return pos


def find_EP_time(pos):
    time_Ep = []
    for i in range(len(pos)):
        if pos[i][1] == 'EP':
            if pos[i][0] in Ep[0]:
                pos[i].append('past')
                time_Ep.append(pos[i][0])
                time_Ep.append(pos[i][3])
            elif pos[i][0] in Ep[1]:
                pos[i].append('present')
                time_Ep.append(pos[i][0])
                time_Ep.append(pos[i][3])
            elif pos[i][0] in Ep[2]:
                pos[i].append('future')
                time_Ep.append(pos[i][0])
                time_Ep.append(pos[i][3])
    for i in range(len(pos)):
        try:
            if pos[i][1] in Predicate and pos[i][1] != 'EP':
                pos[i].append(time_Ep[1])
        except IndexError as E:
            pass
    return pos, time_Ep


def find_predicate(pos):
    tmp_predicate = ''
    tmp = []
    for i in range(len(pos)):
        try:
            if pos[i][2] == 'P':
                if pos[i][1] != 'EP' and pos[i][1] != 'EC' and pos[i][1] != 'EF' and pos[i][1] != 'MAG':
                    tmp_predicate += pos[i][0]
                if pos[i][1] == 'EC' or pos[i][1] == 'EF':
                    if pos[i][0] != '다':
                        pos[i][0] = '다'
                    tmp_predicate += pos[i][0]
                    tmp.append(tmp_predicate)
                    tmp_predicate = ''
                elif pos[i][1] == 'MAG':
                    tmp_predicate += pos[i][0]
                    tmp.append(tmp_predicate)
                    tmp_predicate = ''
        except IndexError as E:
            pass
    if len(tmp_predicate) > 0:
        tmp.append(tmp_predicate)
    return tmp


def EC_separate(pos):
    for i in range(len(pos)):
        if pos[i][1] == 'EC':
            if pos[i][0] == '는다':
                del pos[i]
                nun = ['는', 'EP', 'P']
                da = ['다', 'EC', 'P']
                pos.append(nun)
                pos.append(da)
        elif pos[i][1] in Predicate_plus:
            jaMo = jamos_test(pos[i][0])
            for j in range(len(jaMo) - 2):
                if jaMo[j] == 'ㄴ' and jaMo[j + 1] == '_':
                    tmp = ['ㄴ', 'EP', 'P']
                    pos.append(tmp)
                    del jaMo[j]
            k = 0
            for j in range(len(jaMo)):
                try:
                    if jaMo[j - k] == '_':
                        del jaMo[j - k]
                        k = k + 1
                    elif jaMo[j - k] == ' ':
                        del jaMo[j - k]
                        k = k + 1
                except IndexError as E:
                    pass
            predicate_tmp = join_jamos(jaMo)
            # del pos[i]
            # pre_tmp2 = [predicate_tmp,'VV','P']
            # pos.append(pre_tmp2) # 나는 가지 못했다 => 나 가지 다 못했 으로 변경되어 나와서 아래로 고침ㅇㅇ
            pre_tmp2 = [predicate_tmp, 'VV', 'P']
            pos[i] = pre_tmp2
    return pos


def find_order(pos):
    order = []
    for i in range(len(pos)):
        if pos[i][1] == 'MAG':
            order.append(pos[i][0])
    for i in range(len(pos)):
        if pos[i][1] != 'MAG' and pos[i][1] != 'EP' and pos[i][1] != 'V_process':
            order.append(pos[i][0])

    for i in range(len(pos)):
        if pos[i][1] == 'V_process':
            if len(pos[i][0]) > 1:
                for k in range(len(pos[i][0])):
                    order.append(pos[i][0][k])
            else:
                order.append(pos[i][0][0])
    if '못하다' in order:
        order.remove('못하다')
        order.append('못하다')
    if '하다' in order:
        order.remove('하다')
    if '이다' in order:
        order.remove('이다')
    order1 = str(order)
    return order, order1


def find_nun_eun(pos):
    # pos는 list 형식, 내부 ()는 튜플 형식
    # 튜플을 리스트로 변환(튜플 형식은 append 사용불가)
    for i in range(len(pos)):
        pos[i] = list(pos[i])
    for i in range(len(pos)):
        if pos[i][1] == 'JX':
            if pos[i][0] == '은' or pos[i][0] == '는':
                pos[i][1] = 'JKS'
    return pos


def merge_predicate(pos, predicate, time_ep):
    k = []
    for i in range(len(pos)):
        try:
            if pos[i][2] == 'P':
                k.append(i)
        except IndexError as E:
            pass
    if len(k) > 1:
        tmp = 0
        for j in k:
            del pos[j - tmp]
            tmp += 1
        if time_ep != '':
            t = [predicate, 'V_process', 'P', time_ep]
            pos.append(t)
        else:
            t = [predicate, 'V_process', 'P']
            pos.append(t)
    return pos


def find_negative(pos):
    length = len(pos)
    for j in range(length):
        if pos[j][0] == '하다':
            for i in range(length):
                if pos[i][1] == 'MAG':
                    if pos[i][0] == '못':
                        pos[i][1] = 'V_process'
                        pos[i].append('P')
    for i in range(length):
        if pos[i][1] == 'MAG' and pos[i][2] != 'P':
            if pos[i][0] == '못':
                pos[i][0] = '못하다'
                pos[i].append('P')
    return pos


def change_predicate(predicate, time_ep_only):
    for i in range(len(predicate)):
        if predicate[i] == '않다':
            if time_ep_only == 'past':
                predicate[i] = '아직'
            else:
                predicate[i] = '없다'
    return predicate


# =====================================================================================================================================
# [START speech_transcribe_streaming_mic]

import re
from PyQt5 import QtTest
from google.cloud import speech

import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

import os

credential_path = "C:/Users/user/Downloads/cellular-potion-337013-294a7c0bc513.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QStyle, QSizePolicy, QFileDialog, QHBoxLayout, \
    QVBoxLayout, QLabel, QSlider, QPlainTextEdit
import sys
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtCore import Qt, QUrl


def listen_print_loop(responses):
    speech_text = []
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)

        else:
            print("transcript : " + transcript + "\n")
            if "끝" not in transcript:
                speech_text.append(str(transcript))
                print("speech_text : " + str(speech_text) + "\n")
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(끝|멈춰)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0
    return speech_text


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


# =====================================================================================================================================

# =====================================================================================================================================

# 음성 -> 수화 ui

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.text_box = QPlainTextEdit()
        self.label = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.sendBtn = QPushButton()
        self.transe = QPushButton()
        self.playBtn = QPushButton()
        self.setWindowTitle("음성 -> 수화")  # window 만들기
        self.setGeometry(350, 100, 700, 500)  # window 만들기
        self.setWindowIcon(QIcon('player.png'))  # window 만들기
        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)
        self.init_ui()

        # self.show() # window 만들기

    def init_ui(self):
        # create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videowidget object
        videowidget = QVideoWidget()

        # create butten for playing
        openBtn = QPushButton()
        # 기능 설정
        openBtn.clicked.connect(self.open_file)
        openBtn.setMaximumWidth(60)
        openBtn.setMaximumHeight(25)
        openBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))  # QStyle.SP_MediaVolume
        # create open button

        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        #
        self.transe.setEnabled(True)
        self.transe.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.transe.setMaximumWidth(60)
        self.transe.setMaximumHeight(50)
        # 기능 설정

        self.playBtn.clicked.connect(self.play_video)
        # create send button
        self.sendBtn.setMaximumWidth(60)
        self.sendBtn.setMaximumHeight(25)
        self.sendBtn.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))

        # create slider

        self.slider.setRange(0, 0)
        # 기능
        self.slider.sliderMoved.connect(self.set_position)

        # create label

        self.label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        ##create textBox

        self.text_box.setMaximumHeight(50)
        self.text_box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        # create hbox layout

        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        # set widgets to the hox layout

        vboxLayout2 = QVBoxLayout()
        vboxLayout2.addWidget(openBtn)
        vboxLayout2.addWidget(self.sendBtn)

        # hboxLayout.addWidget(openBtn)

        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)
        # create vbox layout

        # create hbox layout

        hboxLayout2 = QHBoxLayout()
        hboxLayout2.setContentsMargins(0, 0, 0, 0)
        # set widgets to the hox layout

        hboxLayout2.addWidget(self.text_box)
        hboxLayout2.addLayout(vboxLayout2)
        hboxLayout2.addWidget(self.transe)
        # create vbox layout

        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(videowidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addLayout(hboxLayout2)
        vboxLayout.addWidget(self.label)
        self.setLayout(vboxLayout)
        self.mediaPlayer.setVideoOutput(videowidget)

        # media player signals
        # not needed
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        self.sendBtn.clicked.connect(self.open_video)
        self.transe.clicked.connect(self.Home)

    def Home(self):
        window2.start()
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def open_video(self):
        text1 = self.text_box.toPlainText()
        self.komoran_run(text1)
        # filename = self.text_box.toPlainText() + ".wmv" # 확장자는 하나로 통일해서 변경하기.
        # print("in")
        # if filename != '':
        #     self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
        #     self.playBtn.setEnabled(True)
        #     self.play_video()

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.mediaPlayer.play()
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.mediaPlayer.durationChanged.connect(self.time_sleep)

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.set(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())

    def time_sleep(self, duration):
        print(duration)
        QtTest.QTest.qWait(duration * 1000)

    def open_file(self):
        language_code = "ko-KR"  # a BCP-47 language tag

        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            test_sentence = listen_print_loop(responses)
            print(test_sentence)
            test_sentence = str(test_sentence)
            test_sentence = test_sentence.replace("[", "")
            test_sentence = test_sentence.replace("]", "")
            test_sentence = test_sentence.replace("'", "")
            test_sentence = test_sentence.replace(",", "")
            self.komoran_run(test_sentence)
            # self.make_playlist(test_sentence)
            # self.text_box.setPlainText(str(test_sentence))
            # for i in test_sentence:
            #     print(i)
            #     self.text_box.setPlainText(i.strip())
            #     self.open_video()
            #
            #     print("루프 끝")

    def make_playlist(self, test_sentence):
        playlist = QMediaPlaylist(self.mediaPlayer)
        for i in test_sentence:
            filename = i.strip() + ".wmv"
            playlist.addMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            print(i.strip() + 'out')
        self.mediaPlayer.setPlaylist(playlist)
        self.playBtn.setEnabled(True)
        self.mediaPlayer.playlist().setCurrentIndex(0)
        self.mediaPlayer.play()

    def komoran_run(self, text1):
        komoran = Komoran("EXP")
        print("In put: " + text1)
        str1 = komoran.get_plain_text(text1)
        pos = str1.split(' ')
        for i in range(len(pos)):
            pos[i] = pos[i].split('/')
        print("Only komoran: ", end="")
        print(pos)
        pos_original = pos  # pos_original = 형태소 분석기만 사용했을 시 나오는 pos

        pos = find_nun_eun(pos)  # 는이랑 은을 주격 조사로 변경
        # print(pos)
        pos = N_J_processing_in_pos(pos)  # 명사뒤에 조사 붙는걸로 명사 문장성분 파악
        print("Noun_tagging + Remove Josa: ", end="")

        print(pos)
        pos = P_processing_in_text(pos)  # 서술어 파악
        print("predicate: ", end="")
        print(pos)
        pos = time_adverb(pos)  # 시간 부사 파악
        print("time_adv: ", end="")
        print(pos)
        pos = EC_separate(pos)  # 는다 , ㄴ 을 분리( 선어말 어미 분리)

        # print(pos)

        # pos = find_negative(pos) # 못 -> 못하다로 사용

        pos, time_ep = find_EP_time(pos)  # 시제 선어말어미 파악
        # print("time_ep: " + time_ep)
        if len(time_ep) != 0:
            time_ep_only = time_ep[1]  # time_ep 중 time만 파악
        else:
            time_ep_only = ''

        time_ep_only = str(time_ep_only)  # time_ep_only는 선어말 어미에서 가져온 시제를 가진 거
        print("time_ep: ", end="")
        print(time_ep_only)
        predicate = find_predicate(pos)  # 서술어 파악
        # print(predicate)
        predicate = change_predicate(predicate, time_ep_only)  # 서술어 변경 ex) 부정어-> 않다 -> 아직 or 없다
        pos = merge_predicate(pos, predicate, time_ep_only)  # 서술어 합치기
        print("Predicate after processing: ", end="")
        print(pos)
        list_fin, str_fin = find_order(pos)  # 순서 재정렬
        # print(pos) # 문장에서 조사를 뺀 나머지 pos
        # print(predicate) # 서술어 (조사 제외)
        print("Output_string: " + str_fin)  # 순서 정리된 마지막 list
        self.make_playlist(list_fin)
        self.text_box.setPlainText(str(str_fin))


# =====================================================================================================================================
# 수화 -> 음성 ui

running = False


class Window2(QWidget):
    def __init__(self):
        super().__init__()
        self.vbox = QtWidgets.QVBoxLayout()
        self.win = QtWidgets.QWidget()
        self.vbox2 = QtWidgets.QVBoxLayout()
        self.hbox = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel()
        self.text = QtWidgets.QLineEdit()
        self.text.setMinimumHeight(60)
        self.btn_start = QtWidgets.QPushButton("Camera On")
        self.btn_stop = QtWidgets.QPushButton("Camera Off")
        self.btn_switch = QtWidgets.QPushButton()
        # ------
        self.vbox3 = QtWidgets.QVBoxLayout()
        self.btn_rand = QtWidgets.QPushButton("R")
        self.btn_rand2 = QtWidgets.QPushButton("R2")
        # --------
        self.setWindowTitle("음성 -> 수화")  # window 만들기
        self.setWindowIcon(QIcon('player.png'))  # window 만들기
        self.init_ui()
        # self.start()
        # self.show()
        # -------------------------
        # 감사 데이터 부족, 위: 손가락 인식 모누 안됌 존경: 받침 손가락 제대로 인식불가
        self.label_data = ['THANKYOU', 'POLICE', 'HEAD', 'HELLO', 'DOWN', 'UP', 'HOME', 'RESPECT', 'FRIEND', 'DAD', 'STATIC']
        # Actions that we try to detect
        self.actions = np.array(self.label_data)

        self.twoD_list = []
        self.face = [10, 234, 152, 454]
        self.body = range(11, 23)
        self.sequence = []
        self.sentence = []
        self.new_model = keras.models.load_model('sign_language_action_static.h5')

        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16), (16, 117, 245),
                       (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 117, 16), (117, 245, 16)]
        # --------------------------------------------------------------------------------------------------------------
        self.RorR2 = 0
        # --------------------------------------------------------------------------------------------------------------

    def init_ui(self):
        self.btn_start.setMaximumHeight(30)
        self.btn_stop.setMaximumHeight(30)
        self.btn_switch.setMaximumHeight(60)
        self.btn_switch.setMaximumWidth(60)
        self.btn_switch.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.text)
        self.vbox2.addWidget(self.btn_start)
        self.vbox2.addWidget(self.btn_stop)
        # --------------------------------
        self.btn_rand.setMaximumHeight(60)
        self.btn_rand.setMaximumWidth(30)
        self.btn_rand2.setMaximumWidth(30)
        self.btn_rand2.setMaximumHeight(60)
        self.vbox3.addWidget(self.btn_rand)
        self.vbox3.addWidget(self.btn_rand2)
        self.hbox.addLayout(self.vbox3)
        self.btn_rand.clicked.connect(self.set_R1)
        self.btn_rand2.clicked.connect(self.set_R2)
        # ---------------------------------
        self.hbox.addLayout(self.vbox2)
        self.hbox.addWidget(self.btn_switch)
        self.vbox.addLayout(self.hbox)
        self.win.setLayout(self.vbox)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_switch.clicked.connect(self.switch)
        app.aboutToQuit.connect(self.onExit)
        self.setLayout(self.vbox)

    def run(self):
        global running
        global index_num
        index_old = -1
        cap = cv2.VideoCapture(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # ------------------------------------------
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        flag = 0
        # _-----------------------------------------
        # self.label.resize(int(width), int(height))
        self.label.resize(1400, 800)
        if self.RorR2 == 1:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while running:
                    ret, image = cap.read()
                    image = cv2.flip(image, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if ret:
                        image.flags.writeable = False

                        results = holistic.process(image)

                        total_list = pre.extract_landmarks(results)
                        self.sequence.append(total_list)
                        self.sequence = self.sequence[-50:]

                        if len(self.sequence) == 50:
                            twoD_array = pre.set_relative_axial(self.sequence)
                            twoD_array_interpolated = pre.preprocessing_inter(twoD_array, 0)
                            twoD_array_discrete = pre.preprocessing_inter(twoD_array, 1)

                            X_data1 = pre.argumentation(twoD_array, sampling_frame=10)  # (20, 10, 174)? oo
                            X_data2 = pre.argumentation(twoD_array_interpolated, sampling_frame=10)
                            X_data3 = pre.argumentation(twoD_array_discrete, sampling_frame=10)

                            res1 = self.new_model.predict(X_data1)
                            res2 = self.new_model.predict(X_data2)
                            res3 = self.new_model.predict(X_data3)
                            resul = [np.argmax(res1[i]) for i in range(20)] + [np.argmax(res2[i]) for i in
                                                                               range(20)] + [np.argmax(res3[i]) for i in
                                                                                             range(20)]
                            # print(np.bincount(resul))
                            res = np.bincount(resul) / 60

                            # res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
                            self.text.setText(
                                str(self.actions[np.argmax(res)]) + ' ' + str(res[np.argmax(res)] * 100) + " %")

                            # ---초기화---
                            self.sequence = []
                            white1 = np.full((image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
                            alpha = 0.7
                            image = cv2.addWeighted(image, alpha, white1, (1 - alpha), 0)
                            # print to imshow landmark
                            flag = 1

                        image.flags.writeable = True
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        # Draw landmark annotation on the image. holistic
                        image = pre.print_landmarksNindex(image, results.face_landmarks, self.face)
                        image = pre.print_landmarksNindex(image, results.pose_landmarks, self.body)

                        # Draw the hand annotations on the image.
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                        # Flip the image horizontally for a selfie-view display.
                        if flag == 1:
                            image = self.prob_viz(res, self.actions, image, self.colors)

                        image = cv2.resize(image, (1400, 800))
                        h, w, c = image.shape
                        qImg = QtGui.QImage(image.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qImg)
                        self.label.setPixmap(pixmap)

                    else:
                        QtWidgets.QMessageBox.about(self.win, "Error", "Cannot read frame.")
                        print("cannot read frame.")
                        break
        else:
            # 지화 part
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
                while running:
                    ret, image = cap.read()
                    image = cv2.flip(image, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if ret:
                        # Convert the BGR image to RGB before processing.
                        results = hands.process(image)
                        list_of_hand = []
                        if not results.multi_hand_landmarks:
                            image = cv2.resize(image, (1400, 800))
                            h, w, c = image.shape
                            qImg = QtGui.QImage(image.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                            pixmap = QtGui.QPixmap.fromImage(qImg)
                            self.label.setPixmap(pixmap)
                            # cv2.imshow("HandTracking", image)
                            # cv2.waitKey(1)
                            continue
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                list_of_hand = [
                                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
                                ]

                                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        # cv2.imshow("HandTracking", image)
                        # cv2.waitKey(1)

                        list_of_hand = axis_move(list_of_hand)
                        temp = [list_of_hand]
                        # print(temp)

                        # print(*sc.predict(temp))
                        result_signal = signal_detection(sc.predict(temp))
                        sending = ''.join(sc.predict(temp))
                        if signal_detection.signal_times % 15 == 0:
                            print(alph_dic[sending])

                        if result_signal != 0:
                            hangul_result.append(alph_dic[sending])
                            string_result =  '[' + join_jamos(hangul_result) + ']\t\t' + str(hangul_result)
                            # print(hangul_result)
                            # print('[', join_jamos(hangul_result), ']')
                            self.text.setText(string_result)
                            # ===========================================================================
                        image = cv2.resize(image, (1400, 800))
                        h, w, c = image.shape
                        qImg = QtGui.QImage(image.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qImg)
                        self.label.setPixmap(pixmap)
                    else:
                        QtWidgets.QMessageBox.about(self.win, "Error", "Cannot read frame.")
                        print("cannot read frame.")
                        break

        cap.release()
        qImg = QtGui.QImage("black.png")
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)
        # self.label.resize(int(wid), int(hei))
        print("Thread end.")

    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return output_frame

    def stop(self):
        global running
        running = False
        self.text.setText("stoped..")

    def start(self):
        global running
        if running:
            self.text.setText("already running")
            return
        running = True
        th = threading.Thread(target=self.run)
        th.start()
        self.text.setText("started...")

    def onExit(self):
        print("exit")
        self.stop()

    def switch(self):
        self.stop()
        widget.setCurrentIndex(widget.currentIndex() - 1)

    # ============================================================================================================

    def set_R1(self):
        self.RorR2 = 1

    def set_R2(self):
        self.RorR2 = 2


# =====================================================================================================================================
# main part
index_word = ["감사", "경찰", "머리", "안녕", "아래", "위", "집", "존경", "친구", "아버지"]

app = QtWidgets.QApplication([])
widget = QtWidgets.QStackedWidget()
window = Window()
window2 = Window2()
widget.addWidget(window)
widget.addWidget(window2)
widget.setFixedHeight(1000)
widget.setFixedWidth(1450)
widget.setWindowTitle("한국어 수화 번역기")
widget.show()
sys.exit(app.exec_())
