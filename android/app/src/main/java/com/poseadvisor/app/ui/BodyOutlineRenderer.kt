package com.poseadvisor.app.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import com.poseadvisor.app.pose.PoseModel
import com.poseadvisor.app.pose.jx
import com.poseadvisor.app.pose.jy
import kotlin.math.sqrt

/**
 * 参数化解剖人形渲染器：
 * 以 25 关节为骨架，按人体解剖宽度剖面（肩/肘/腕/髋/膝/踝各段宽度）
 * 沿骨骼法向偏移生成轮廓控制点，贝塞尔平滑为闭合的真人形剪影。
 * 支持任意姿势（含 AI 合成姿势）实时变形，无需任何图片资源。
 */
object BodyOutlineRenderer {

    data class Palette(
        val fillColor: Int,
        val strokeColor: Int,
    )

    // ---- 解剖宽度剖面（单位：躯干长度 骨盆→颈）----
    private const val HEAD_RX = 0.34f   // 头宽半径
    private const val HEAD_RY = 0.44f   // 头高半径
    private const val NECK_W = 0.13f    // 颈半宽
    private const val UARM_W1 = 0.17f   // 上臂根粗
    private const val UARM_W2 = 0.12f   // 肘粗
    private const val FARM_W2 = 0.08f   // 腕粗
    private const val HAND_LEN = 0.16f  // 手掌长
    private const val TORSO_PAD = 0.05f // 胸廓外扩
    private const val WAIST_RATIO = 0.80f // 腰相对肩宽比例
    private const val HIP_PAD = 0.09f   // 髋外扩
    private const val THIGH_W1 = 0.20f  // 大腿根粗
    private const val CALF_W1 = 0.13f   // 膝粗
    private const val CALF_W2 = 0.075f  // 踝粗
    private const val FOOT_LEN = 0.20f  // 脚长

    // ================= 引导剪影（实心人形） =================

    /**
     * 绘制丰满人形轮廓。
     * @param pose 像素坐标 75 维向量
     * @param torsoRef 躯干长度（像素）
     */
    fun draw(canvas: Canvas, pose: FloatArray, torsoRef: Float, palette: Palette) {
        val t = torsoRef.coerceAtLeast(1f)
        val pts = buildOutlinePoints(pose, t)
        val path = smoothClosedPath(pts)

        val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = palette.fillColor
            style = Paint.Style.FILL
        }
        canvas.drawPath(path, fill)
        if (palette.strokeColor != 0) {
            val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = palette.strokeColor
                style = Paint.Style.STROKE
                strokeWidth = t * 0.045f + 1f
                strokeJoin = Paint.Join.ROUND
            }
            canvas.drawPath(path, stroke)
        }
    }

    // ================= 用户实时人体（分段着色反馈） =================

    /**
     * 用户人体：同样的解剖轮廓思路，但按肢体段分组着色，
     * 用于"达标青 / 偏差橙"的实时反馈。
     * @param limbColors 6 个颜色：左上臂/左前臂/右上臂/右前臂/左腿/右腿
     */
    fun drawUser(
        canvas: Canvas,
        pose: FloatArray,
        torsoRef: Float,
        torsoColor: Int,
        headColor: Int,
        limbColors: IntArray,
    ) {
        val t = torsoRef.coerceAtLeast(1f)

        // 四肢：渐细胶囊（圆头粗线），按段着色（含髋踝中点估计的膝盖）
        drawUserLimbs(canvas, pose, t, limbColors)

        // 躯干：肩-髋梯形
        val torsoPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = torsoColor
            style = Paint.Style.FILL
        }
        canvas.drawPath(torsoPath(pose, t), torsoPaint)

        // 头
        val headPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = headColor
            style = Paint.Style.FILL
        }
        canvas.drawCircle(pose.jx(PoseModel.HEAD), pose.jy(PoseModel.HEAD), HEAD_RX * t, headPaint)
    }

    private fun drawUserLimbs(canvas: Canvas, pose: FloatArray, t: Float, limbColors: IntArray) {
        val kneeL = kneeOf(pose, true)
        val kneeR = kneeOf(pose, false)
        // (ax,ay,bx,by,widthRatio,colorIdx)
        val segs = listOf(
            floatArrayOf(pose.jx(PoseModel.SHOULDER_L), pose.jy(PoseModel.SHOULDER_L), pose.jx(PoseModel.ELBOW_L), pose.jy(PoseModel.ELBOW_L), UARM_W1, 0f),
            floatArrayOf(pose.jx(PoseModel.ELBOW_L), pose.jy(PoseModel.ELBOW_L), pose.jx(PoseModel.WRIST_L), pose.jy(PoseModel.WRIST_L), FARM_W2 * 1.6f, 1f),
            floatArrayOf(pose.jx(PoseModel.SHOULDER_R), pose.jy(PoseModel.SHOULDER_R), pose.jx(PoseModel.ELBOW_R), pose.jy(PoseModel.ELBOW_R), UARM_W1, 2f),
            floatArrayOf(pose.jx(PoseModel.ELBOW_R), pose.jy(PoseModel.ELBOW_R), pose.jx(PoseModel.WRIST_R), pose.jy(PoseModel.WRIST_R), FARM_W2 * 1.6f, 3f),
            floatArrayOf(pose.jx(PoseModel.HIP_L), pose.jy(PoseModel.HIP_L), kneeL[0], kneeL[1], THIGH_W1, 4f),
            floatArrayOf(kneeL[0], kneeL[1], pose.jx(PoseModel.ANKLE_L), pose.jy(PoseModel.ANKLE_L), CALF_W1, 4f),
            floatArrayOf(pose.jx(PoseModel.HIP_R), pose.jy(PoseModel.HIP_R), kneeR[0], kneeR[1], THIGH_W1, 5f),
            floatArrayOf(kneeR[0], kneeR[1], pose.jx(PoseModel.ANKLE_R), pose.jy(PoseModel.ANKLE_R), CALF_W1, 5f),
        )
        for (s in segs) {
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = limbColors[s[5].toInt().coerceIn(0, limbColors.size - 1)]
                style = Paint.Style.STROKE
                strokeWidth = s[4] * t
                strokeCap = Paint.Cap.ROUND
            }
            canvas.drawLine(s[0], s[1], s[2], s[3], paint)
        }
    }

    private fun torsoPath(pose: FloatArray, t: Float): Path {
        val pad = TORSO_PAD * t
        return Path().apply {
            moveTo(pose.jx(PoseModel.SHOULDER_L) - pad, pose.jy(PoseModel.SHOULDER_L))
            lineTo(pose.jx(PoseModel.SHOULDER_R) + pad, pose.jy(PoseModel.SHOULDER_R))
            lineTo(pose.jx(PoseModel.HIP_R) + pad, pose.jy(PoseModel.HIP_R))
            lineTo(pose.jx(PoseModel.HIP_L) - pad, pose.jy(PoseModel.HIP_L))
            close()
        }
    }

    // ================= 轮廓控制点生成 =================

    private fun norm(x: Float, y: Float): FloatArray {
        val l = sqrt(x * x + y * y)
        return if (l < 1e-6f) floatArrayOf(0f, -1f) else floatArrayOf(x / l, y / l)
    }

    private fun perp(d: FloatArray) = floatArrayOf(-d[1], d[0])

    private fun dot(a: FloatArray, b: FloatArray) = a[0] * b[0] + a[1] * b[1]

    private fun pt(pose: FloatArray, j: Int, dx: Float, dy: Float) =
        floatArrayOf(pose.jx(j) + dx, pose.jy(j) + dy)

    /** 膝盖估计：髋踝中点（25 关节格式无膝盖） */
    private fun kneeOf(pose: FloatArray, left: Boolean): FloatArray {
        val hip = if (left) PoseModel.HIP_L else PoseModel.HIP_R
        val ankle = if (left) PoseModel.ANKLE_L else PoseModel.ANKLE_R
        return floatArrayOf(
            (pose.jx(hip) + pose.jx(ankle)) / 2,
            (pose.jy(hip) + pose.jy(ankle)) / 2,
        )
    }

    /** 生成闭合人形轮廓控制点（从头右侧顺时针） */
    private fun buildOutlinePoints(pose: FloatArray, t: Float): List<FloatArray> {
        val pts = ArrayList<FloatArray>(48)

        val bodyUp = norm(
            pose.jx(PoseModel.NECK) - pose.jx(PoseModel.PELVIS),
            pose.jy(PoseModel.NECK) - pose.jy(PoseModel.PELVIS)
        )
        val sideL = perp(bodyUp) // 体侧方向（画面左）

        val head = floatArrayOf(pose.jx(PoseModel.HEAD), pose.jy(PoseModel.HEAD))
        val pelvis = floatArrayOf(pose.jx(PoseModel.PELVIS), pose.jy(PoseModel.PELVIS))

        val halfShoulder = (distOf(pose, PoseModel.SHOULDER_L, PoseModel.SHOULDER_R) / 2)
        val halfHip = (distOf(pose, PoseModel.HIP_L, PoseModel.HIP_R) / 2)

        // ---- 头部（右→顶→左）----
        val headRx = HEAD_RX * t
        val headRy = HEAD_RY * t
        fun headPt(u: Float, l: Float) = floatArrayOf(
            head[0] + bodyUp[0] * u + sideL[0] * l,
            head[1] + bodyUp[1] * u + sideL[1] * l
        )
        val neckR = pt(pose, PoseModel.NECK, -sideL[0] * NECK_W * t, -sideL[1] * NECK_W * t)
        val neckL = pt(pose, PoseModel.NECK, sideL[0] * NECK_W * t, sideL[1] * NECK_W * t)

        // ---- 手臂链 ----
        fun armChain(shoulder: Int, elbow: Int, wrist: Int, outSign: Float): List<FloatArray> {
            val s = floatArrayOf(pose.jx(shoulder), pose.jy(shoulder))
            val e = floatArrayOf(pose.jx(elbow), pose.jy(elbow))
            val w = floatArrayOf(pose.jx(wrist), pose.jy(wrist))
            val u1 = norm(e[0] - s[0], e[1] - s[1])
            var n1 = perp(u1)
            // 法向朝外：与体侧方向同号
            if (dot(n1, floatArrayOf(sideL[0] * outSign, sideL[1] * outSign)) < 0) {
                n1 = floatArrayOf(-n1[0], -n1[1])
            }
            val u2 = norm(w[0] - e[0], w[1] - e[1])
            var n2 = perp(u2)
            if (dot(n2, n1) < 0) n2 = floatArrayOf(-n2[0], -n2[1])

            val handTip = floatArrayOf(w[0] + u2[0] * HAND_LEN * t, w[1] + u2[1] * HAND_LEN * t)
            return listOf(
                floatArrayOf(s[0] + n1[0] * UARM_W1 * t, s[1] + n1[1] * UARM_W1 * t),      // 肩外
                floatArrayOf(e[0] + n1[0] * UARM_W2 * t, e[1] + n1[1] * UARM_W2 * t),      // 肘外
                floatArrayOf(w[0] + n2[0] * FARM_W2 * t, w[1] + n2[1] * FARM_W2 * t),      // 腕外
                handTip,                                                                    // 指尖
                floatArrayOf(w[0] - n2[0] * FARM_W2 * t, w[1] - n2[1] * FARM_W2 * t),      // 腕内
                floatArrayOf(e[0] - n1[0] * UARM_W2 * t, e[1] - n1[1] * UARM_W2 * t),      // 肘内
                floatArrayOf(s[0] - n1[0] * UARM_W1 * t * 0.75f, s[1] - n1[1] * UARM_W1 * t * 0.75f), // 腋窝
            )
        }

        // ---- 腿链 ----
        fun legChain(hip: Int, ankle: Int, outSign: Float): List<FloatArray> {
            val h = floatArrayOf(pose.jx(hip), pose.jy(hip))
            val a = floatArrayOf(pose.jx(ankle), pose.jy(ankle))
            val k = floatArrayOf((h[0] + a[0]) / 2, (h[1] + a[1]) / 2)
            val u3 = norm(k[0] - h[0], k[1] - h[1])
            var n3 = perp(u3)
            if (dot(n3, floatArrayOf(sideL[0] * outSign, sideL[1] * outSign)) < 0) {
                n3 = floatArrayOf(-n3[0], -n3[1])
            }
            val u4 = norm(a[0] - k[0], a[1] - k[1])
            var n4 = perp(u4)
            if (dot(n4, n3) < 0) n4 = floatArrayOf(-n4[0], -n4[1])

            // 脚尖：沿小腿方向 + 略外撇
            val toeDir = norm(u4[0] + n3[0] * 0.5f, u4[1] + n3[1] * 0.5f)
            val toe = floatArrayOf(a[0] + toeDir[0] * FOOT_LEN * t, a[1] + toeDir[1] * FOOT_LEN * t)

            return listOf(
                floatArrayOf(h[0] + n3[0] * THIGH_W1 * t, h[1] + n3[1] * THIGH_W1 * t),     // 髋外
                floatArrayOf(k[0] + n3[0] * CALF_W1 * t, k[1] + n3[1] * CALF_W1 * t),       // 膝外
                floatArrayOf(a[0] + n4[0] * CALF_W2 * t, a[1] + n4[1] * CALF_W2 * t),       // 踝外
                toe,                                                                         // 脚尖
                floatArrayOf(a[0] - n4[0] * CALF_W2 * t, a[1] - n4[1] * CALF_W2 * t),       // 踝内
                floatArrayOf(k[0] - n3[0] * CALF_W1 * t, k[1] - n3[1] * CALF_W1 * t),       // 膝内
            )
        }

        // ---- 躯干侧边（腋窝→髋）----
        fun torsoSide(outSign: Float): List<FloatArray> {
            val sL = floatArrayOf(sideL[0] * outSign, sideL[1] * outSign)
            val chest = pt(pose, PoseModel.CHEST, sL[0] * (halfShoulder + TORSO_PAD * t), sL[1] * (halfShoulder + TORSO_PAD * t))
            val waist = pt(pose, PoseModel.SPINE_MID, sL[0] * halfShoulder * WAIST_RATIO, sL[1] * halfShoulder * WAIST_RATIO)
            return listOf(chest, waist)
        }

        val armL = armChain(PoseModel.SHOULDER_L, PoseModel.ELBOW_L, PoseModel.WRIST_L, +1f)
        val armR = armChain(PoseModel.SHOULDER_R, PoseModel.ELBOW_R, PoseModel.WRIST_R, -1f)
        val legL = legChain(PoseModel.HIP_L, PoseModel.ANKLE_L, +1f)
        val legR = legChain(PoseModel.HIP_R, PoseModel.ANKLE_R, -1f)
        val torsoL = torsoSide(+1f)
        val torsoR = torsoSide(-1f)

        val crotchL = floatArrayOf(pelvis[0] + sideL[0] * halfHip * 0.25f, pelvis[1] + 0.06f * t)
        val crotchR = floatArrayOf(pelvis[0] - sideL[0] * halfHip * 0.25f, pelvis[1] + 0.06f * t)

        // 头右侧 → 右臂外 → 右手 → 右臂内 → 右躯干 → 右腿外 → 右脚 → 右腿内 →
        // 裆 → 左腿内 → 左脚 → 左腿外 → 左躯干 → 左臂内 → 左手 → 左臂外 → 头左侧 → 头顶
        pts += neckR
        pts += armR.subList(0, 4)          // 肩外→肘外→腕外→指尖
        pts += armR.subList(4, 7)          // 腕内→肘内→腋窝
        pts += torsoR
        pts += floatArrayOf(pose.jx(PoseModel.HIP_R) - sideL[0] * HIP_PAD * t, pose.jy(PoseModel.HIP_R) - sideL[1] * HIP_PAD * t)
        pts += legR
        pts += crotchR
        pts += crotchL
        pts += legL.asReversed()
        pts += floatArrayOf(pose.jx(PoseModel.HIP_L) + sideL[0] * HIP_PAD * t, pose.jy(PoseModel.HIP_L) + sideL[1] * HIP_PAD * t)
        pts += torsoL.asReversed()
        pts += armL.subList(6, 7)          // 腋窝
        pts += armL.subList(4, 6).asReversed() // 肘内→腕内
        pts += armL.subList(3, 4)          // 指尖
        pts += armL.subList(0, 3).asReversed() // 腕外→肘外→肩外
        pts += neckL
        // 头部五点弧（比三点更圆润，避免尖顶）
        pts += headPt(0f, headRx)                       // 头左
        pts += headPt(headRy * 0.72f, headRx * 0.72f)   // 左上
        pts += headPt(headRy, 0f)                        // 头顶
        pts += headPt(headRy * 0.72f, -headRx * 0.72f)  // 右上
        pts += headPt(0f, -headRx)                       // 头右

        return pts
    }

    /** 中点平滑法生成闭合贝塞尔路径 */
    private fun smoothClosedPath(pts: List<FloatArray>): Path {
        val n = pts.size
        fun mid(a: FloatArray, b: FloatArray) = floatArrayOf((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        val path = Path()
        val m0 = mid(pts[0], pts[1])
        path.moveTo(m0[0], m0[1])
        for (i in 1 until n) {
            val m = mid(pts[i], pts[(i + 1) % n])
            path.quadTo(pts[i][0], pts[i][1], m[0], m[1])
        }
        val mLast = mid(pts[n - 1], pts[0])
        path.quadTo(pts[0][0], pts[0][1], mLast[0], mLast[1])
        path.close()
        return path
    }

    private fun distOf(p: FloatArray, a: Int, b: Int): Float {
        val dx = p.jx(b) - p.jx(a); val dy = p.jy(b) - p.jy(a)
        return sqrt(dx * dx + dy * dy)
    }

    // ================= 缩略图 =================

    /** 渲染姿势缩略图（实心人形剪影，供卡片显示） */
    fun renderThumbnail(pose75: FloatArray, sizePx: Int, palette: Palette, background: Int): Bitmap {
        val bmp = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(background)

        val keyIdx = intArrayOf(0, 3, 4, 5, 10, 11, 13, 14, 16, 17, 12, 15, 7, 8)
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in keyIdx) {
            minX = minOf(minX, pose75.jx(i)); maxX = maxOf(maxX, pose75.jx(i))
            minY = minOf(minY, pose75.jy(i)); maxY = maxOf(maxY, pose75.jy(i))
        }
        val cx = (minX + maxX) / 2; val cy = (minY + maxY) / 2
        val span = maxOf(maxX - minX, maxY - minY) * 1.3f + 1e-4f
        val scale = sizePx / span

        val px = FloatArray(PoseModel.POSE_DIM)
        for (i in 0 until PoseModel.NUM_JOINTS) {
            px[i * 3] = (pose75.jx(i) - cx) * scale + sizePx / 2f
            px[i * 3 + 1] = (pose75.jy(i) - cy) * scale + sizePx / 2f
        }
        draw(canvas, px, distOf(px, PoseModel.PELVIS, PoseModel.NECK), palette)
        return bmp
    }
}
