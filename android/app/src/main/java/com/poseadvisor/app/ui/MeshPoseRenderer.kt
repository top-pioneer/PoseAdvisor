package com.poseadvisor.app.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffColorFilter
import android.graphics.RectF
import com.poseadvisor.app.pose.PoseComparator
import com.poseadvisor.app.pose.PoseModel
import com.poseadvisor.app.pose.jx
import com.poseadvisor.app.pose.jy
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 真人剪影姿势渲染器：
 * 内置公有领域真人剪影（Wikimedia Commons, Public Domain），
 * 用 Moving Least Squares（Schaefer 2006, affine）把剪影网格
 * 变形到目标姿势，经 Canvas.drawBitmapMesh 渲染。
 * 引导人形 = 真实人体轮廓，而非几何拼接。
 */
object MeshPoseRenderer {

    private const val MESH_W = 7          // drawBitmapMesh 横向格数
    private const val MESH_H = 11         // 纵向格数
    private const val MLS_EPS = 900f      // 权重平滑（像素平方），防近锚点爆炸

    private var bitmap: Bitmap? = null
    private var basePts = FloatArray(0)   // 锚点基坐标（位图像素）N*2
    private var ruleA = IntArray(0)
    private var ruleB = IntArray(0)
    private var ruleS = FloatArray(0)
    private var ruleR = FloatArray(0)
    private var srcGrid = FloatArray(0)   // 源网格顶点（位图坐标，行优先）

    @Volatile
    private var initialized = false

    fun init(context: Context) {
        if (initialized) return
        synchronized(this) {
            if (initialized) return
            bitmap = BitmapFactory.decodeStream(context.assets.open("silhouette.png"))
            val text = context.assets.open("silhouette_anchors.json").bufferedReader().use { it.readText() }
            val arr = JSONObject(text).getJSONArray("anchors")
            val n = arr.length()
            basePts = FloatArray(n * 2)
            ruleA = IntArray(n); ruleB = IntArray(n)
            ruleS = FloatArray(n); ruleR = FloatArray(n)
            val bw = bitmap!!.width
            val bh = bitmap!!.height
            for (i in 0 until n) {
                val a = arr.getJSONObject(i)
                val base = a.getJSONArray("base")
                basePts[i * 2] = base.getDouble(0).toFloat() * bw
                basePts[i * 2 + 1] = base.getDouble(1).toFloat() * bh
                val rule = a.getJSONArray("rule")
                ruleA[i] = rule.getInt(0)
                ruleB[i] = rule.getInt(1)
                ruleS[i] = rule.getDouble(2).toFloat()
                ruleR[i] = rule.getDouble(3).toFloat()
            }
            // 源网格（行优先，(MESH_H+1) 行 (MESH_W+1) 列）
            val verts = FloatArray((MESH_W + 1) * (MESH_H + 1) * 2)
            var k = 0
            for (r in 0..MESH_H) {
                for (c in 0..MESH_W) {
                    verts[k++] = c * (bw - 1).toFloat() / MESH_W
                    verts[k++] = r * (bh - 1).toFloat() / MESH_H
                }
            }
            srcGrid = verts
            initialized = true
        }
    }

    /** 锚点目标位置（归一化姿势空间，x 已镜像对齐剪影标注方向） */
    private fun anchorTargets(pose: FloatArray): FloatArray {
        val n = ruleA.size
        val out = FloatArray(n * 2)
        for (i in 0 until n) {
            val ai = ruleA[i]; val bi = ruleB[i]
            val ax = pose.jx(ai); val ay = pose.jy(ai)
            val dx = pose.jx(bi) - ax; val dy = pose.jy(bi) - ay
            val len = sqrt(dx * dx + dy * dy)
            if (len < 1e-6f) {
                out[i * 2] = ax; out[i * 2 + 1] = ay
            } else {
                val px = -dy / len; val py = dx / len
                out[i * 2] = ax + dx * ruleS[i] + px * ruleR[i]
                out[i * 2 + 1] = ay + dy * ruleS[i] + py * ruleR[i]
            }
        }
        return out
    }

    /** Schaefer MLS-affine：把源网格顶点变形到目标锚点定义的姿势 */
    private fun mlsWarp(tgt: FloatArray): FloatArray {
        val nVerts = srcGrid.size / 2
        val n = basePts.size / 2
        val out = FloatArray(srcGrid.size)
        val w = FloatArray(n)
        for (k in 0 until nVerts) {
            val vx = srcGrid[k * 2]; val vy = srcGrid[k * 2 + 1]
            var wsum = 0f
            var psx = 0f; var psy = 0f; var qsx = 0f; var qsy = 0f
            for (i in 0 until n) {
                val dx = basePts[i * 2] - vx
                val dy = basePts[i * 2 + 1] - vy
                val wi = 1f / (dx * dx + dy * dy + MLS_EPS)
                w[i] = wi
                wsum += wi
                psx += wi * basePts[i * 2]; psy += wi * basePts[i * 2 + 1]
                qsx += wi * tgt[i * 2]; qsy += wi * tgt[i * 2 + 1]
            }
            psx /= wsum; psy /= wsum; qsx /= wsum; qsy /= wsum
            // M = (Σw Ph^T Ph)^{-1} · (Σw Ph^T Qh)
            var m2a = 0f; var m2b = 0f; var m2c = 0f
            var m1a = 0f; var m1b = 0f; var m1c = 0f; var m1d = 0f
            for (i in 0 until n) {
                val wi = w[i]
                val px = basePts[i * 2] - psx; val py = basePts[i * 2 + 1] - psy
                val qx = tgt[i * 2] - qsx; val qy = tgt[i * 2 + 1] - qsy
                m2a += wi * px * px; m2b += wi * px * py; m2c += wi * py * py
                m1a += wi * px * qx; m1b += wi * px * qy
                m1c += wi * py * qx; m1d += wi * py * qy
            }
            // 2x2 逆矩阵
            val det = m2a * m2c - m2b * m2b + 1e-8f
            val i2a = m2c / det; val i2b = -m2b / det; val i2c = m2a / det
            val ma = i2a * m1a + i2b * m1c
            val mb = i2a * m1b + i2b * m1d
            val mc = i2b * m1a + i2c * m1c
            val md = i2b * m1b + i2c * m1d
            val rx = vx - psx; val ry = vy - psy
            out[k * 2] = rx * ma + ry * mc + qsx
            out[k * 2 + 1] = rx * mb + ry * md + qsy
        }
        return out
    }

    // 变形结果缓存：引导姿势不常变，避免每帧重复 MLS 计算
    private var cachedPoseRef: FloatArray? = null
    private var cachedRectKey = 0
    private var cachedVerts: FloatArray? = null

    /**
     * 将姿势（75 维原始坐标）绘制到目标矩形区域。
     * @param destRect 位图完整帧在画布上的位置（保持位图宽高比由调用方决定）
     */
    fun draw(
        canvas: Canvas,
        pose75: FloatArray,
        destRect: RectF,
        tintColor: Int,
        alpha: Int = 0xB0,
    ) {
        if (!initialized) return
        val bmp = bitmap ?: return

        val rectKey = destRect.hashCode()
        val verts = if (cachedPoseRef === pose75 && cachedRectKey == rectKey && cachedVerts != null) {
            cachedVerts!!
        } else {
            // 归一化 + 比例矫正 + 镜像（对齐剪影标注方向）
            val pose = PoseComparator.normalize(pose75)
            for (i in 0 until PoseModel.NUM_JOINTS) pose[i * 3] = -pose[i * 3]

            val tgt = anchorTargets(pose)
            // 目标锚点（躯干单位）→ 目标矩形内像素
            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            for (i in 0 until tgt.size / 2) {
                minX = min(minX, tgt[i * 2]); maxX = max(maxX, tgt[i * 2])
                minY = min(minY, tgt[i * 2 + 1]); maxY = max(maxY, tgt[i * 2 + 1])
            }
            val spanY = (maxY - minY).coerceAtLeast(1e-3f)
            val scale = destRect.height() * 0.94f / spanY
            val cx = (minX + maxX) / 2; val cy = (minY + maxY) / 2
            val dcx = destRect.centerX(); val dcy = destRect.centerY()
            for (i in 0 until tgt.size / 2) {
                tgt[i * 2] = (tgt[i * 2] - cx) * scale + dcx
                tgt[i * 2 + 1] = (tgt[i * 2 + 1] - cy) * scale + dcy
            }
            val v = mlsWarp(tgt)
            cachedPoseRef = pose75
            cachedRectKey = rectKey
            cachedVerts = v
            v
        }

        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            colorFilter = PorterDuffColorFilter(tintColor, PorterDuff.Mode.SRC_IN)
            this.alpha = alpha
        }
        canvas.drawBitmapMesh(bmp, MESH_W, MESH_H, verts, 0, null, 0, paint)
    }

    /** 位图宽高比（w/h），未初始化返回 0 */
    fun bitmapAspect(): Float {
        val b = bitmap ?: return 0f
        return if (b.height > 0) b.width.toFloat() / b.height else 0f
    }

    /** 渲染姿势缩略图（供卡片显示） */
    fun renderThumbnail(context: Context, pose75: FloatArray, sizePx: Int, tintColor: Int, bgColor: Int): Bitmap {
        init(context)
        val bmp = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(bgColor)
        val margin = sizePx * 0.04f
        draw(canvas, pose75, RectF(margin, margin, sizePx - margin, sizePx - margin), tintColor, 0xFF)
        return bmp
    }
}
