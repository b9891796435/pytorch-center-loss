import threading
import time
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import imutils
import jwt
import cv2
import numpy as np
import pymysql.cursors
import aiohttp_cors
from aiohttp import web
from real_time_video import frame_parse

from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust",
            "fear", "contempt"]
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
# 数据库连接
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             database='project',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
# jwt密钥及函数
jwt_secret = 'twilight-sparkle'


def encodeJwt(Obj):
    return jwt.encode(Obj, jwt_secret, 'HS256')


def decodeJwt(Str):
    return jwt.decode(Str, jwt_secret, ['HS256'])


async def login(request):  # 登录
    params = await request.json()
    with connection.cursor() as cursor:

        try:
            sql = 'select * from user where uname="%s" and password="%s"' % (
                params['username'], params['password'])
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                raise Exception('Exc')
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'success', 'token': encodeJwt(
                        {'userid': result['USERID'], 'permission': result['AUT_NUM']}), 'permission': result['AUT_NUM']}
                ), )
        except Exception:
            print(Exception)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'fail', 'msg': '用户名或密码错误'}
                ), )


async def register(request):  # 注册
    params = await request.json()
    with connection.cursor() as cursor:

        try:
            sql = 'select * from user where uname="%s"' % (params['username'])
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                sql = 'insert into user ' \
                      '(name,age,sex,uname,password,aut_num) ' \
                      'values ("%s",%d,"%s","%s","%s",0)' % (
                          'a_soul', 18, 'M', params['username'], params['password'],
                      )
                cursor.execute(sql)
                connection.commit()
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'success', 'token': encodeJwt(
                            {'userid': params['USERID'], 'permission': 0}),
                         'permission': 0}
                    ), )
            else:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({'status': 'fail', 'msg': '用户名已使用'}))

        except Exception:
            print(Exception)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'fail'}
                ), )


async def profile(request):  # 个人信息
    params = await request.json()
    try:
        user = decodeJwt(request.headers['auth'])
        with connection.cursor() as cursor:
            try:
                sql = 'select * from user where userid=%d' % user['userid']
                cursor.execute(sql)
                result = cursor.fetchone()
                if result is None:
                    raise Exception('Exc')
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        'status': 'success',
                        'data': {
                            'profile': result,
                        }
                    }))
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def queryQueryRecord(request):  # 查询“查询记录”
    params = await request.json()
    try:
        user = decodeJwt(request.headers['auth'])
        with connection.cursor() as cursor:
            try:
                sql = 'select * from query_record where userid=%d limit 10 offset %d' \
                      % (params['userid'], params['page']*10)
                cursor.execute(sql)
                result = cursor.fetchall()
                sql = 'select count(*) from query_record where userid=%d' % params['userid']
                cursor.execute(sql)
                count = cursor.fetchone()['count(*)']
                if result is None:
                    raise Exception('Exc')
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        'status': 'success',
                        'data': {
                            'record': result,
                            'count': count
                        }
                    }))
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def superProfile(request):  # 超管查询用户信息
    params = await request.json()
    try:
        user = decodeJwt(request.headers['auth'])
        if user['permission'] != 2:
            raise Exception()
        with connection.cursor() as cursor:
            sql = 'select *,CAST(sum(COUNT) as UNSIGNED) as DONATION,u.USERID as USERID from user u ' \
                  'left join donate_tbl dt on u.USERID = dt.USERID ' \
                  'GROUP BY u.USERID limit 10 offset %d' % (params['page']*10)
            cursor.execute(sql)
            result = cursor.fetchall()
            sql = 'select count(*) from user'
            cursor.execute(sql)
            count = cursor.fetchone()['count(*)']
            if result is None:
                raise Exception('Exc')
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    'status': 'success',
                    'data': {
                        'count': count,
                        'profile': result,
                    }
                }))
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def cancelAccount(request):  # 注销账号
    params = None
    try:
        params = await request.json()
    except Exception:
        pass
    try:
        user = decodeJwt(request.headers['auth'])
        with connection.cursor() as cursor:
            try:
                uid = params['userid'] if params is not None and 'userid' in params and user['permission'] == 2 else user['userid']
                sql = 'delete from user where USERID=%d' % (uid)
                cursor.execute(sql)
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
        connection.commit()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'success'}
            ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def deleteDynamic(request):  # 删除动态
    params = None
    try:
        params = await request.json()
    except Exception:
        pass
    try:
        user = decodeJwt(request.headers['auth'])
        if user['permission'] < 1:
            raise Exception()
        with connection.cursor() as cursor:
            try:
                sql = 'delete from dynamic where DYNAMIC_ID=%d' % params['dynamicID']
                cursor.execute(sql)
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
        connection.commit()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'success'}
            ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def profileRecord(request):  # 个人信息页查询使用记录，分页
    params = await request.json()
    try:
        user = decodeJwt(request.headers['auth'])
        with connection.cursor() as cursor:
            try:
                sql = 'select * from USE_RECORD where userid=%d ' \
                      'order by USE_TIME desc limit 10 offset %d' \
                      % (user['userid'], params['page']*10)
                cursor.execute(sql)
                result = cursor.fetchall()
                sql = 'select count(*) from USE_RECORD where userid=%d' \
                      % user['userid']
                cursor.execute(sql)
                result2 = cursor.fetchone()['count(*)']
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        'status': 'success',
                        'data': {
                            'record': result,
                            'count': result2
                        }
                    }))
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def queryRecord(request):  # 管理员查询使用记录，分页
    params = await request.json()
    try:
        user = decodeJwt(request.headers['auth'])
        if (user['permission'] < 1):
            raise Exception()
        with connection.cursor() as cursor:
            try:
                sql = 'select * from use_record where USERID = %d ' \
                      'order by USE_TIME desc limit 10 offset %d' % (
                          params['userId'], params['page'] * 10)
                cursor.execute(sql)
                result = cursor.fetchall()
                sql = 'select count(*) from use_record where USERID = %d' % params['userId']
                cursor.execute(sql)
                count = cursor.fetchone()
                sql = 'insert into query_record (USERID, QUERYID, QUERYTIME) ' \
                      'values (%d,%d,%d)' % (
                          user['userid'], params['userId'], round(time.time())
                      )
                cursor.execute(sql)
                connection.commit()
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"count": count['count(*)'], "data": result}))
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def editProfile(request):  # 更改个人信息及提权
    params = await request.json()
    try:
        user = decodeJwt(request.headers['auth'])
        with connection.cursor() as cursor:
            try:
                uid = params['userid'] if 'userid' in params and user['permission'] == 2 else user['userid']
                aut = ',AUT_NUM=%d' % params['autNum'] if 'autNum' in params and user['permission'] == 2 else ''
                sql = 'update user set NAME="%s",AGE=%d,SEX="%s",' \
                      'UNAME="%s",PASSWORD="%s"%s where USERID=%d' \
                      % (params['NAME'], params['AGE'], params['SEX'],
                         params['UNAME'], params['PASSWORD'],
                         aut, uid)
                cursor.execute(sql)
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
        connection.commit()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'success'}
            ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def pubDynamic(request):  # 发布动态
    params = await request.json()
    with connection.cursor() as cursor:

        try:
            user = decodeJwt(request.headers["auth"])
            sql = 'insert into dynamic ' \
                  '(PUBLISH_TIME,DY_CONTENT,RECORD_ID,USERID) ' \
                  'values (%d,"%s",%d,%d)' % (
                      round(time.time()), params['content'],
                      params['recordId'], user['userid']
                  )
            cursor.execute(sql)
            connection.commit()
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'success'}
                ), )

        except Exception:
            print(Exception)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'fail'}
                ), )


async def donation(request):  # 支持本站
    params = await request.json()
    with connection.cursor() as cursor:
        try:
            user = decodeJwt(request.headers["auth"])
            sql = 'insert into donate_tbl (COUNT, USERID) VALUES (%d,%d)' \
                  % (params['count'], user['userid'])
            cursor.execute(sql)
            connection.commit()
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'success'}
                ), )
        except Exception:
            print(Exception)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'fail'}
                ), )


async def pubComment(request):  # 发布评论
    params = await request.json()
    with connection.cursor() as cursor:
        try:
            user = decodeJwt(request.headers["auth"])
            sql = 'insert into comment ' \
                  '(PUBLISH_TIME,COMMENT_CONTENT,FORWARD_ID,USERID) ' \
                  'values (%d,"%s",%d,%d)' % (
                      round(time.time()), params['content'],
                      params['dynamicID'], user['userid']
                  )
            cursor.execute(sql)
            connection.commit()
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'success'}
                ), )
        except Exception:
            print(Exception)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'fail'}
                ), )


async def likeComment(request):  # 点赞评论
    params = await request.json()
    with connection.cursor() as cursor:

        try:
            user = decodeJwt(request.headers["auth"])
            sql = 'select * from comment_like_tbl where COMMENT_ID=%d and USERID=%d'\
                  % (params['commentID'], user['userid'])
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                user = decodeJwt(request.headers["auth"])
                sql = 'insert into comment_like_tbl (COMMENT_ID, USERID) VALUES (%d,%d)' \
                      % (params['commentID'], user['userid'])
                cursor.execute(sql)
                connection.commit()
                sql = 'update comment set LIKE_COUNT = LIKE_COUNT+1 where COMMENT_ID=%d' \
                      % (params['commentID'])
                cursor.execute(sql)
                connection.commit()
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'success'}
                    ), )
            else:
                raise Exception()

        except Exception:
            print(Exception)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'fail'}
                ), )


async def likeDynamic(request):  # 点赞动态
    params = await request.json()
    with connection.cursor() as cursor:

        try:
            user = decodeJwt(request.headers["auth"])
            sql = 'select * from like_tbl where DYNAMIC_ID=%d and USERID=%d'\
                  % (params['dynamicID'], user['userid'])
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                user = decodeJwt(request.headers["auth"])
                sql = 'insert into like_tbl (DYNAMIC_ID, USERID) VALUES (%d,%d)' \
                      % (params['dynamicID'], user['userid'])
                cursor.execute(sql)
                connection.commit()
                sql = 'update dynamic set LIKE_COUNT = LIKE_COUNT+1 where DYNAMIC_ID=%d' \
                      % (params['dynamicID'])
                cursor.execute(sql)
                connection.commit()
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'success'}
                    ), )
            else:
                raise Exception("temp")

        except Exception:
            print(Exception)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {'status': 'fail'}
                ), )


async def dynamicList(request):  # 动态列表
    try:
        user = decodeJwt(request.headers['auth'])
        with connection.cursor() as cursor:
            try:
                sql = 'SELECT DYNAMIC_ID,NAME,PUBLISH_TIME,DY_CONTENT,LIKE_COUNT,PIC_PATH FROM dynamic dy ' \
                      'inner join user u on dy.USERID = u.USERID ' \
                      'inner join use_record ur on dy.RECORD_ID = ur.RECORD_ID ' \
                      'limit 10 offset %d' % (int(request.query['page'])*10)
                cursor.execute(sql)
                result = cursor.fetchall()
                sql = 'SELECT count(*) FROM dynamic'
                cursor.execute(sql)
                count = cursor.fetchone()['count(*)']
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"status": "success", "data": result, "count": count}))
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def dynamicDetail(request):  # 动态详情
    params = await request.json()
    try:
        user = decodeJwt(request.headers['auth'])
        with connection.cursor() as cursor:
            try:
                sql = 'SELECT COMMENT_ID,NAME,PUBLISH_TIME,COMMENT_CONTENT,LIKE_COUNT FROM comment com ' \
                      'inner join user u on com.USERID = u.USERID where FORWARD_ID = %d' \
                      ' limit 10 offset %d'\
                      % (params['dynamicID'], params['page']*10)
                cursor.execute(sql)
                result = cursor.fetchall()
                sql = 'SELECT count(*) from comment where FORWARD_ID=%d' \
                      % (params['dynamicID'])
                cursor.execute(sql)
                count = cursor.fetchone()['count(*)']
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"status": "success", "data": result, 'count': count})
                )
            except Exception:
                print(Exception)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {'status': 'fail'}
                    ), )
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )


async def offer(request):  # WebRTC握手请求及对接收视频的处理
    params = await request.json()
    try:
        user = decodeJwt(params['token'])
    except Exception:
        print(Exception)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'status': 'fail'}
            ), )
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    predict = None

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            nonlocal predict  # 利用datachannel将识别结果传至前端
            if predict is not None:
                channel.send(json.dumps({'data': predict.tolist()}))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            recorder.addTrack(track)
        elif track.kind == "video":
            class VideoTransformTrack(MediaStreamTrack):
                """
                A video stream track that transforms frames from an another track.
                """

                kind = "video"

                def __init__(self, track):
                    super().__init__()  # don't forget this!
                    self.track = track

                async def recv(self):  # 提取帧，进行识别并标注人脸位置
                    frame = await self.track.recv()
                    frame_array = frame.to_ndarray(format="bgr24")
                    frame_array = imutils.resize(frame_array, width=300)
                    frame_clone = np.copy(frame_array)

                    result = frame_parse(frame_array)
                    if result is not None:
                        (preds, faces) = result
                        (fX, fY, fW, fH) = faces
                        nonlocal predict
                        predict = preds
                        label = EMOTIONS[preds.argmax()]
                    else:
                        return frame

                    cv2.putText(frame_clone, label, (fX, fY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)
                    frame_clone = VideoFrame.from_ndarray(
                        frame_clone, format="bgr24")
                    print(frame.pts, frame.time_base, frame.time)
                    frame_clone.pts = frame.pts + 270000
                    frame_clone.time_base = frame.time_base
                    return frame_clone

            localTrack = VideoTransformTrack(
                relay.subscribe(track)
            )
            pc.addTrack(localTrack)
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():  # RTC链接结束时将最后一次识别的结果与时间存入数据库
            log_info("Track %s ended", track.kind)
            with connection.cursor() as cursor:
                nonlocal user, predict
                temp = []
                for i in predict.tolist():
                    temp.append(round(i, 2))
                sql = 'insert into use_record (USERID, USE_TIME, PIC_PATH) values (%d,%d,"%s")' % (
                    user['userid'], round(time.time()), str(temp).replace(', ', '.'))
                cursor.execute(sql)
            connection.commit()
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="Port for HTTP server (default: 8081)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    # 路由表
    app.router.add_post("/login", login)
    app.router.add_post("/register", register)
    app.router.add_post("/offer", offer)
    app.router.add_post("/profile", profile)
    app.router.add_post("/profileRecord", profileRecord)
    app.router.add_post("/editProfile", editProfile)
    app.router.add_post("/queryRecord", queryRecord)
    app.router.add_post("/queryQueryRecord", queryQueryRecord)
    app.router.add_post("/pubDynamic", pubDynamic)
    app.router.add_post("/pubComment", pubComment)
    app.router.add_post("/dynamicDetail", dynamicDetail)
    app.router.add_post("/deleteDynamic", deleteDynamic)
    app.router.add_post("/likeComment", likeComment)
    app.router.add_post("/likeDynamic", likeDynamic)
    app.router.add_post("/superProfile", superProfile)
    app.router.add_post("/donation", donation)
    app.router.add_post("/cancelAccount", cancelAccount)
    app.router.add_get("/dynamicList", dynamicList)
    cors = aiohttp_cors.setup(app, defaults={  # 配置CORS
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
