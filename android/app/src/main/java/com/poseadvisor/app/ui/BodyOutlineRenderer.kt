package com.poseadvisor.app.ui

import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import com.poseadvisor.app.pose.PoseModel
import com.poseadvisor.app.pose.jx
import com.poseadvisor.app.pose.jy

/**
 * 用户实时人体渲染器：
 * 将被摄者的实时姿态绘制为实心人体（渐细胶囊四肢 + 躯干梯形 + 头部），
 * 按肢体段着色（达标青 / 偏差橙），用于匹配反馈。
 * 引导姿势的真人剪影渲染见 MeshPoseRenderer。
 */
object BodyOutlineRenderer {

    // 肢体粗细剖面（单位：躯干长度 骨盆→颈）
    private const val HEAD_RX = 0.34f
    private const val UARM_W1 = 0.17f   // 上臂
    private const val FARM_W2 = 0.13f   // 前臂
    private const val THIGH_W1 = 0.20f  // 大腿
    private const val CALF_W1 = 0.13f   // 小腿
    private const val TORSO_PAD = 0.05f // 躯干外扩

    /**
     * 绘制用户人体。
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
        drawUserLimbs(canvas, pose, t, limbColors)

        val torsoPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = torsoColor
            style = Paint.Style.FILL
        }
        canvas.drawPath(torsoPath(pose, t), torsoPaint)

        val headPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = headColor
            style = Paint.Style.FILL
        }
        canvas.drawCircle(pose.jx(PoseModel.HEAD), pose.jy(PoseModel.HEAD), HEAD_RX * t, headPaint)
    }

    private fun drawUserLimbs(canvas: Canvas, pose: FloatArray, t: Float, limbColors: IntArray) {
        val kneeL = kneeOf(pose, true)
        val kneeR = kneeOf(pose, false)
        // (ax, ay, bx, by, widthRatio, colorIdx)
        val segs = listOf(
            floatArrayOf(pose.jx(PoseModel.SHOULDER_L), pose.jy(PoseModel.SHOULDER_L), pose.jx(PoseModel.ELBOW_L), pose.jy(PoseModel.ELBOW_L), UARM_W1, 0f),
            floatArrayOf(pose.jx(PoseModel.ELBOW_L), pose.jy(PoseModel.ELBOW_L), pose.jx(PoseModel.WRIST_L), pose.jy(PoseModel.WRIST_L), FARM_W2, 1f),
            floatArrayOf(pose.jx(PoseModel.SHOULDER_R), pose.jy(PoseModel.SHOULDER_R), pose.jx(PoseModel.ELBOW_R), pose.jy(PoseModel.ELBOW_R), UARM_W1, 2f),
            floatArrayOf(pose.jx(PoseModel.ELBOW_R), pose.jy(PoseModel.ELBOW_R), pose.jx(PoseModel.WRIST_R), pose.jy(PoseModel.WRIST_R), FARM_W2, 3f),
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

    /** 膝盖估计：髋踝中点（25 关节格式无膝盖） */
    private fun kneeOf(pose: FloatArray, left: Boolean): FloatArray {
        val hip = if (left) PoseModel.HIP_L else PoseModel.HIP_R
        val ankle = if (left) PoseModel.ANKLE_L else PoseModel.ANKLE_R
        return floatArrayOf(
            (pose.jx(hip) + pose.jx(ankle)) / 2,
            (pose.jy(hip) + pose.jy(ankle)) / 2,
        )
    }
}
