# 4.2 SSAO

## 理论

### 基础

环境光遮蔽（AO：Ambient Occlusion） 模拟光线与物体的遮挡关系，SS即Screen Space，指利用屏幕空间信息（像素深度、法线等缓冲）实现物体在间接光下的阴影。

### 算法原理

![image-20210830144435693](https://i.loli.net/2021/08/30/3wjntoZHdNBxTA8.png)

深度缓冲：重构相机（观察）空间中坐标（Z）重构该视点下三维场景

![image-20210830145032137](https://i.loli.net/2021/08/30/QMcu5lUCf41x38e.png)

法线缓冲：重构每像素“法线-切线-副法线”构成的坐标轴，用以计算法线半球采样的**随机向量**（描述该像素AO强度）

![image-20210830145044249](https://i.loli.net/2021/08/30/Ll6UqvnRstCxk4M.png)

法向半球

黑：**当前**计算样本 	蓝：样本法向量	白、灰：采样点（灰色：被遮挡点，深度大于周围，据此判断最终AO强度）

若为全球形采样会导致墙面遮挡强度过高，因为样本周围一半采样点都在表面下（被遮挡）

![image-20210830151304739](https://i.loli.net/2021/08/30/vZVPMUtl3JK821m.png)

## 算法实现

获取深度&法线缓存数据

C#:

```C#
//获取深度&法线缓存数据
private void Start()
{
    cam = this.GetComponent<Camera>();
    //相机渲染模式为带深度和法线
    cam.depthTextureMode = cam.depthTextureMode | DepthTextureMode.DepthNormals;
}
```

shader:

```c
//获取深度法线图
sampler2D _CameraDepthNormalsTexture;

//...other code

//采样获得深度值和法线值
float3 viewNormal;
float linear01Depth;
float4 depthnormal = tex2D(_CameraDepthNormalsTexture, i.uv); //uv属于屏幕空间（当前渲染图像）
DecodeDepthNormal(depthnormal, linear01Depth, viewNormal);    //解码数据，获取采样后深度值和法线值

//...other code

inline void DecodeDepthNormal( float4 enc, out float depth, out float3 normal )
{
    depth = DecodeFloatRG (enc.zw);
    normal = DecodeViewNormalStereo (enc);
}
```



### 改进

## 其他AO方案

