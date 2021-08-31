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

### 获取深度&法线缓存数据

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

inline void DecodeDepthNormal( float4 enc, out float depth, out float3 normal )//in UnityCG.cginc 
{
    depth = DecodeFloatRG (enc.zw);
    normal = DecodeViewNormalStereo (enc);
}
```

### 重建相机空间坐标

参考：[Unity从深度缓冲重建世界空间位置](https://zhuanlan.zhihu.com/p/92315967)

从[NDC](https://zhuanlan.zhihu.com/p/65969162)（标准化设备坐标）空间重建

![image-20210830160737698](https://i.loli.net/2021/08/30/vPQ2VEtriwUkKWo.png)

```c
//step1:计算样本屏幕坐标 vertex shader
float4 screenPos = ComputeScreenPos(o.vertex);//屏幕纹理坐标 
//step2: 转换至NDC空间 vertex shader
float4 ndcPos = (screenPos / screenPos.w) * 2 - 1;// NDC position
//step3: 计算相机空间中至远屏幕方向 vertex shader
float3 clipVec = float3(ndcPos.x, ndcPos.y, 1.0) * _ProjectionParams.z; //_ProjectionParams.z -> 相机远平面
//step4:矩阵变换至相机空间中样本相对相机的方向
o.viewVec = mul(unity_CameraInvProjection, clipVec.xyzz).xyz;
//step5:重建相机空间样本坐标 fragment shader
float3 viewPos = linear01Depth * i.viewVec;//获取像素相机屏幕坐标位置

/*
屏幕空间->NDC空间->裁剪空间-逆投影矩阵->观察（相机）空间
*/
```

### 构建法向量正交基

![image-20210830195602464](https://i.loli.net/2021/08/30/DnRfS2CUAs647eY.png)

```C
//Step1 设置法向量
//获取像素相机屏幕法线，法相z方向相对于相机为负（所以 需要乘以-1置反），并处理成单位向量
viewNormal = normalize(viewNormal) * float3(1, 1, -1);
//Step2 randvec法线半球的随机向量(用于构建随机的正交基，而非所有样本正交基一致)，此处先设置为统一变量（后面优化会改成随机）
float3 randvec = normalize(float3(1,1,1));
//Step3 求切向量 利用函数cross叉积求负切向量
/*
Gramm-Schimidt处理创建正交基
法线&切线&副切线构成的坐标空间
*/
float3 tangent = normalize(randvec - viewNormal * dot(randvec, viewNormal));
float3 bitangent = cross(viewNormal, tangent);
float3x3 TBN = float3x3(tangent, bitangent, viewNormal);
```

关于TBN：[ 切线空间（TBN） ---- 聊聊图形学中的矩阵运算](https://blog.csdn.net/chishanxu3325/article/details/100858834)	unity入门精要4.7 法线变换

![image-20210830203753569](https://i.loli.net/2021/08/30/yCk1q2nGSv3I4Dp.png)

![image-20210830210311596](https://i.loli.net/2021/08/30/BAV7632wdWvHemx.png)

### AO采样

C# 生成随机样本

```c#
private void GenerateAOSampleKernel()
{
//...other code...
    for (int i = 0; i < SampleKernelCount; i++) //在此生成随机样本
    {
        var vec = new Vector4(Random.Range(-1.0f, 1.0f), Random.Range(-1.0f, 1.0f), Random.Range(0, 1.0f), 1.0f);
        vec.Normalize();
        var scale = (float)i / SampleKernelCount;
        //使分布符合二次方程的曲线
        scale = Mathf.Lerp(0.01f, 1.0f, scale * scale);
        vec *= scale;
        sampleKernelList.Add(vec);
    }
}
```

shader 比较法线半球中样本深度与观察点深度以确定AO强度

```c
for (int i = 0; i < sampleCount; i++)
{
    //随机向量，转化至法线切线空间中 得到此法线半球的随机向量
    float3 randomVec = mul(_SampleKernelArray[i].xyz, TBN);

    //ao权重
    float weight = smoothstep(0, 0.2, length(randomVec.xy));
    
    //计算随机法线半球后的向量
    float3 randomPos = viewPos + randomVec * _SampleKeneralRadius;
    //转换到屏幕坐标
    float3 rclipPos = mul((float3x3)unity_CameraProjection, randomPos);
    float2 rscreenPos = (rclipPos.xy / rclipPos.z) * 0.5 + 0.5;
/*
观察（相机）空间->-投影矩阵->裁剪空间->屏幕空间
*/
    
    float randomDepth;
    float3 randomNormal;
    float4 rcdn = tex2D(_CameraDepthNormalsTexture, rscreenPos);
    DecodeDepthNormal(rcdn, randomDepth, randomNormal);

    //采样点的深度值和样本深度比对前后关系
    ao += (randomDepth>=linear01Depth)?1.0:0.0;//是否有遮挡关系???【存疑】
    //认为判断条件应该为(randomDepth<=linear01Depth)，如果随机样本离相机近，应该深度小
}
```

![image-20210830215815116](https://i.loli.net/2021/08/30/ApfTeJ9qPUWVl74.png)

![image-20210830221124849](https://i.loli.net/2021/08/30/LgtuxlAMj4VfJnm.png)

### 改进

## 其他AO方案

