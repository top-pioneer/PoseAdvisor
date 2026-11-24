package com.poseadvisor.app.pose

import android.content.Context
import org.json.JSONObject

/** 姿势库中的一条姿势记录 */
data class PoseEntry(
    val id: String,
    val category: String,
    val label: String,
    val thumbnail: String,
    val joints: FloatArray,   // 75 维，归一化图像坐标
)

data class PoseCategory(val id: String, val label: String)

/** 从 assets 加载离线生成的姿势库（VAE 生成，PC 端 tools/generate_pose_library.py 产出） */
class PoseLibrary private constructor(
    val categories: List<PoseCategory>,
    val poses: List<PoseEntry>,
) {
    fun posesOf(categoryId: String): List<PoseEntry> = poses.filter { it.category == categoryId }

    companion object {
        fun load(context: Context): PoseLibrary {
            val text = context.assets.open("pose_library.json").bufferedReader().use { it.readText() }
            val root = JSONObject(text)

            val categories = mutableListOf<PoseCategory>()
            val cats = root.getJSONArray("categories")
            for (i in 0 until cats.length()) {
                val c = cats.getJSONObject(i)
                categories.add(PoseCategory(c.getString("id"), c.getString("label")))
            }

            val poses = mutableListOf<PoseEntry>()
            val arr = root.getJSONArray("poses")
            for (i in 0 until arr.length()) {
                val p = arr.getJSONObject(i)
                val jarr = p.getJSONArray("joints")
                val flat = FloatArray(PoseModel.POSE_DIM)
                for (jt in 0 until PoseModel.NUM_JOINTS) {
                    val jp = jarr.getJSONArray(jt)
                    flat[jt * 3] = jp.getDouble(0).toFloat()
                    flat[jt * 3 + 1] = jp.getDouble(1).toFloat()
                    flat[jt * 3 + 2] = jp.getDouble(2).toFloat()
                }
                poses.add(
                    PoseEntry(
                        id = p.getString("id"),
                        category = p.getString("category"),
                        label = p.getString("label"),
                        thumbnail = p.getString("thumbnail"),
                        joints = flat,
                    )
                )
            }
            return PoseLibrary(categories, poses)
        }
    }
}
