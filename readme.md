# 4.2 SSAO

项目地址：[Graphic-researcher/UnityCrytekSponza-master2019 (github.com)](https://github.com/Graphic-researcher/UnityCrytekSponza-master2019)

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

	//判断累加ao值
	//采样点的深度值和样本深度比对前后关系
	//ao += (randomDepth>=linear01Depth)?1.0:0.0;//是否有遮挡关//老师这里可能笔误了
	ao += (randomDepth>=linear01Depth)?0.0:1.0;//是否有遮挡关系
}
```

![image-20210830215815116](https://i.loli.net/2021/08/30/ApfTeJ9qPUWVl74.png)

![image-20210830221124849](https://i.loli.net/2021/08/30/LgtuxlAMj4VfJnm.png)

当前版本shader：

```c
//https://blog.csdn.net/tianhai110/article/details/5684128?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-7.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-7.channel_param
Shader "ImageEffect/SSAO0"
{
    Properties
    {
        [HideInInspector]_MainTex ("Texture", 2D) = "white" {}
    }

	CGINCLUDE
    #include "UnityCG.cginc"
	struct appdata
    {
        float4 vertex : POSITION;
        float2 uv : TEXCOORD0;
    };

    struct v2f
    {
        float2 uv : TEXCOORD0;
        float4 vertex : SV_POSITION;
		float3 viewVec : TEXCOORD1;
		float3 viewRay : TEXCOORD2;
    };

	#define MAX_SAMPLE_KERNEL_COUNT 64

	float3 _randomVec;
	bool _isRandom;

	sampler2D _MainTex;
	//获取深度法线图
	sampler2D _CameraDepthNormalsTexture;
    
	//Ao
	sampler2D _NoiseTex;
	float4 _SampleKernelArray[MAX_SAMPLE_KERNEL_COUNT];
	float _SampleKernelCount;
	float _SampleKeneralRadius;
	float _DepthBiasValue;
	float _RangeStrength;
	float _AOStrength;
    v2f vert_Ao (appdata v)
    {
        v2f o;
        o.vertex = UnityObjectToClipPos(v.vertex);
        o.uv = v.uv;
		
		//计算相机空间中的像素方向（相机到像素的方向）
		//https://zhuanlan.zhihu.com/p/92315967
		//屏幕纹理坐标
		float4 screenPos = ComputeScreenPos(o.vertex);
		// NDC position 转换至NDC空间
		float4 ndcPos = (screenPos / screenPos.w) * 2 - 1;
		// 计算至远屏幕方向
		float3 clipVec = float3(ndcPos.x, ndcPos.y, 1.0) * _ProjectionParams.z;//_ProjectionParams.z -> 相机远平面
		//矩阵变换至相机空间中样本相对相机的方向
		o.viewVec = mul(unity_CameraInvProjection, clipVec.xyzz).xyz;

		/*
		屏幕空间->NDC空间->裁剪空间-逆投影矩阵->观察（相机）空间
		*/
        return o;
    }

	//Ao计算
    fixed4 frag_Ao (v2f i) : SV_Target
    {
        //采样屏幕纹理
        fixed4 col = tex2D(_MainTex, i.uv);

		//采样获得深度值和法线值
		float3 viewNormal;
		float linear01Depth;
		float4 depthnormal = tex2D(_CameraDepthNormalsTexture,i.uv);//uv属于屏幕空间（当前渲染图像）
		DecodeDepthNormal(depthnormal,linear01Depth,viewNormal);//解码数据，获取采样后深度值和法线值

		//获取像素相机屏幕坐标位置
		float3 viewPos = linear01Depth * i.viewVec;

		//获取像素相机屏幕法线，法相z方向相对于相机为负（所以 需要乘以-1置反），并处理成单位向量
		viewNormal = normalize(viewNormal) * float3(1, 1, -1);

		//铺平纹理
		float2 noiseScale = _ScreenParams.xy / 4.0;
		float2 noiseUV = i.uv * noiseScale;
		float3 randvec = normalize(float3(1,1,1));
		//randvec法线半球的随机向量
		if(_isRandom)
			randvec = normalize(_randomVec);
		else
			randvec = tex2D(_NoiseTex,noiseUV).xyz;
		//float3 randvec = normalize(float3(1,1,1));
		//float3 randvec = tex2D(_NoiseTex,noiseUV).xyz;
		//Gramm-Schimidt处理创建正交基
		//法线&切线&副切线构成的坐标空间
		float3 tangent = normalize(randvec - viewNormal * dot(randvec,viewNormal));//求切向量 
		float3 bitangent = cross(viewNormal,tangent);//利用函数cross叉积求负切向量
		float3x3 TBN = float3x3(tangent,bitangent,viewNormal);

		//采样核心
		float ao = 0;
		int sampleCount = _SampleKernelCount;//每个像素点上的采样次数
		//https://blog.csdn.net/qq_39300235/article/details/102460405
		for(int i=0;i<sampleCount;i++){
			//随机向量，转化至法线切线空间中 得到此法线半球（TBN空间）的随机向量
			float3 randomVec = mul(_SampleKernelArray[i].xyz,TBN);
			
			//ao权重
			float weight = smoothstep(0,0.2,length(randomVec.xy));
			
			//计算随机法线半球后的向量
			float3 randomPos = viewPos + randomVec * _SampleKeneralRadius;
			//转换到屏幕坐标
			float3 rclipPos = mul((float3x3)unity_CameraProjection, randomPos);
			float2 rscreenPos = (rclipPos.xy / rclipPos.z) * 0.5 + 0.5;

			float randomDepth;
			float3 randomNormal;
			float4 rcdn = tex2D(_CameraDepthNormalsTexture, rscreenPos);
            //随机样本在屏幕的深度和法线信息
			DecodeDepthNormal(rcdn, randomDepth, randomNormal);
			
			//判断累加ao值
			//采样点的深度值和样本深度比对前后关系
			ao += (randomDepth>=linear01Depth)?1.0:0.0;
		}

		ao = ao/sampleCount;
		return float4(ao,ao,ao,1);
    }
	
	//应用AO贴图
	
	sampler2D _AOTex;

	fixed4 frag_Composite(v2f i) : SV_Target
	{
		fixed4 col = tex2D(_MainTex, i.uv);
		fixed4 ao = tex2D(_AOTex, i.uv);
		col.rgb *= ao.r;
		return col;
	}

	ENDCG

    SubShader
    {	
		Cull Off ZWrite Off ZTest Always

		//Pass 0 : Generate AO 
		Pass
        {
            CGPROGRAM
            #pragma vertex vert_Ao
            #pragma fragment frag_Ao
            ENDCG
        }
        //Pass 1 : Composite AO
		Pass
		{
			CGPROGRAM
			#pragma vertex vert_Ao
			#pragma fragment frag_Composite
			ENDCG
		}
    }
}

```

### 当前AO_Shader执行图：

(randomDepth>=linear01Depth)?1.0:0.0



![image-20210904151720677](https://i.loli.net/2021/09/04/l8fcPBU69gjDuFC.png)

![image-20210904153809164](https://i.loli.net/2021/09/04/KgtsZL2B9IihDXJ.png)

如下视频，更改随机向量，会看到遮挡关系随之变化

<video src=".\Vedio\randomVec.mp4"></video>

如下视频，更改采样点数量以及采样半径，遮挡关系的变化，采样的样本越多越好，但是采样半径需要适中，过小效果不明显，过大又过于强烈

<video src=".\Vedio\Sample.mp4"></video>

## 改进

### 增加随机性

上述算法基本概况了SSAO算法的情况，但还有很多优化空间。

首先，我们的随机向量固定太死，结果可能会很生硬，所以我们可以对噪声贴图采样以增加正交基随机性

```css
//铺平纹理
float2 noiseScale = _ScreenParams.xy / 4.0;
float2 noiseUV = i.uv * noiseScale;
float3 randvec = normalize(float3(1,1,1));
randvec = tex2D(_NoiseTex,noiseUV).xyz;
```

<img src="https://i.loli.net/2021/09/04/KfR2HG1xUmwiJDy.png" alt="noise" style="zoom:1000%;" />

![image-20210904165239706](https://i.loli.net/2021/09/04/azRcYbIVoOps4md.png)

### AO累加平滑

#### 范围判断（模型边界）

如下图，在AO图中，天空和屋顶形成遮蔽关系，而这种情况不符合现实

![image-20210904165816812](https://i.loli.net/2021/09/04/oIb7Xx6p1HAJZnB.png)

![image-20210904165844884](https://i.loli.net/2021/09/04/YrBn6AEHtOxD7Gm.png)

原因是如果随机样本在屏幕上对应为背景（天空）区域，深度值和当前着色点相差很大，可能会导致错误的遮挡关系，再看下图，龙不该挡住远处的建筑，这也是SSAO经典问题

![image-20210906221419081](https://i.loli.net/2021/09/06/kC5TqXyvPcuarDO.png)

```c
float range = abs(randomDepth - linear01Depth) > _RangeStrength ? 0.0 : 1.0;//解决深度差过大（模型边界）
```

![image-20210904201955772](https://i.loli.net/2021/09/04/Zoef6uASPT93san.png)

此时模型边缘阴影问题缓解了

![image-20210906221841484](https://i.loli.net/2021/09/06/NsPUV1y54DMrBm9.png)

![image-20210904202544334](https://i.loli.net/2021/09/04/nuEScXNeFQgMvdP.png)

#### 自阴影判定

上图我们发现在一些墙面上出现了阴影，而同一平面不会有遮挡关系，这是由于随机点深度值和着色点深度很近，我们可以增加深度偏移

![image-20210906220558754](https://i.loli.net/2021/09/06/3omgaQNIuWEHFTb.png)

![image-20210906220745365](https://i.loli.net/2021/09/06/AGzTjBupc4MLQn9.png)

```c
float selfCheck = randomDepth + _DepthBiasValue < linear01Depth ? 1.0 : 0.0;//解决自阴影
```

![image-20210904203953932](https://i.loli.net/2021/09/04/awpJ4fekA2LcroS.png)

![image-20210904204029173](https://i.loli.net/2021/09/04/KDFSaqxc35WnmjZ.png)

![image-20210906221059618](https://i.loli.net/2021/09/06/duLiUcsjyqQlV2n.png)

![image-20210906221014720](https://i.loli.net/2021/09/06/cQt67PTXCaID92w.png)

深度值的把控也是一种取舍，如上较大的偏移虽然解决部分墙面自阴影问题，但龙上的遮蔽细节大打折扣

#### AO权重平滑

```c
//ao权重 
float weight = smoothstep(0,0.2,length(randomVec.xy));
```

![image-20210904204214717](https://i.loli.net/2021/09/04/yRTm7rqnkoGYteP.png)

#### 模糊AO贴图

双边滤波或其他滤波方式

```c
	fixed4 frag_Blur (v2f i) : SV_Target
	{
		//_MainTex_TexelSize -> https://forum.unity.com/threads/_maintex_texelsize-whats-the-meaning.110278/
		float2 delta = _MainTex_TexelSize.xy * _BlurRadius.xy;
		
		float2 uv = i.uv;
		float2 uv0a = i.uv - delta;
		float2 uv0b = i.uv + delta;	
		float2 uv1a = i.uv - 2.0 * delta;
		float2 uv1b = i.uv + 2.0 * delta;
		float2 uv2a = i.uv - 3.0 * delta;
		float2 uv2b = i.uv + 3.0 * delta;
		
		float3 normal = GetNormal(uv);
		float3 normal0a = GetNormal(uv0a);
		float3 normal0b = GetNormal(uv0b);
		float3 normal1a = GetNormal(uv1a);
		float3 normal1b = GetNormal(uv1b);
		float3 normal2a = GetNormal(uv2a);
		float3 normal2b = GetNormal(uv2b);
		
		fixed4 col = tex2D(_MainTex, uv);
		fixed4 col0a = tex2D(_MainTex, uv0a);
		fixed4 col0b = tex2D(_MainTex, uv0b);
		fixed4 col1a = tex2D(_MainTex, uv1a);
		fixed4 col1b = tex2D(_MainTex, uv1b);
		fixed4 col2a = tex2D(_MainTex, uv2a);
		fixed4 col2b = tex2D(_MainTex, uv2b);
		
		half w = 0.37004405286;
		half w0a = CompareNormal(normal, normal0a) * 0.31718061674;
		half w0b = CompareNormal(normal, normal0b) * 0.31718061674;
		half w1a = CompareNormal(normal, normal1a) * 0.19823788546;
		half w1b = CompareNormal(normal, normal1b) * 0.19823788546;
		half w2a = CompareNormal(normal, normal2a) * 0.11453744493;
		half w2b = CompareNormal(normal, normal2b) * 0.11453744493;
		
		half3 result;
		result = w * col.rgb;
		result += w0a * col0a.rgb;
		result += w0b * col0b.rgb;
		result += w1a * col1a.rgb;
		result += w1b * col1b.rgb;
		result += w2a * col2a.rgb;
		result += w2b * col2b.rgb;
		
		result /= w + w0a + w0b + w1a + w1b + w2a + w2b;
		return fixed4(result, 1.0);
	}
```

![image-20210904204445674](https://i.loli.net/2021/09/04/HpkOgmsV6TB7ydz.png)

## 对比

法线

![image-20210906215343099](https://i.loli.net/2021/09/06/2wO8lhieAdbJEt6.png)

关闭SSAO（包含lightmap）

![image-20210904204743764](https://i.loli.net/2021/09/04/2Tp95EDAuBOY3HC.png)

开启SSAO

![image-20210904204838364](https://i.loli.net/2021/09/04/W6V32iKSNajgFBh.png)

## 复杂场景里的SSAO

参数设置（未模糊AO），有两次开关AO操作

![image-20210908142538887](https://i.loli.net/2021/09/08/z9sp8jLb4qDBI5f.png)

<video src=".\Vedio\SSAO漫游.mp4"></video>

<video src=".\Vedio\SSAO滤波漫游.mp4"></video>

滤波后AO会平滑一些（这里个人看了一些文章后有一些观点，SSAO是工业上的一种近似，所以输出的物理结果并不会十分理想（不像离线渲染是实实在在地计算光线和物体），滤波这种消除噪声的操作也可以理解为缓解错误结果的办法）

![image-20210908143456583](https://i.loli.net/2021/09/08/asV7UJRIqdPlMOX.png)

![image-20210908143536071](https://i.loli.net/2021/09/08/OPGRnoD47ca9E8x.png)

## Unity Post Processing AO

Unity后处理组件中的后处理组件对SSAO进行了增强，有Scalable Ambient Obscurance和Multi Scale Volumetric Obscurance，两者相对SSAO的法线半球采样采用了更优方法，具体可阅读下方链接。

学习链接：[Scalable Ambient Obscurance - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/202622927)

[shader复习与深入:Screen Space Ambient Occlusion(屏幕空间环境光遮蔽)-腾讯游戏学院 (qq.com)](https://gameinstitute.qq.com/community/detail/108633)

官方源码地址：[PostProcessing/PostProcessing/Shaders/Builtins at v2 · Unity-Technologies/PostProcessing (github.com)](https://github.com/Unity-Technologies/PostProcessing/tree/v2/PostProcessing/Shaders/Builtins)

官方文档：[Ambient Occlusion | Package Manager UI website (unity3d.com)](https://docs.unity3d.com/Packages/com.unity.postprocessing@2.1/manual/Ambient-Occlusion.html)

说明：下方输出AO的方式采用**Post-Process Debug**组件

![image-20210908215239355](https://i.loli.net/2021/09/08/fz9CcUZj5GrtVKe.png)

### Scalable Ambient Obscurance

过强的AO强度会导致带状区域的出现（这在接下来讨论的几个AO方法中也会出现）

![image-20210908221447809](https://i.loli.net/2021/09/08/U91MxLjBPmZoFY4.png)

过大的AO半径则会导致严重错误的遮挡关系

![image-20210908222206849](https://i.loli.net/2021/09/08/6vReLgam8fT97XG.png)

![image-20210908222219794](https://i.loli.net/2021/09/08/ZK4MbsxF8OnpvlI.png)

调节较为合适的参数如下：

![image-20210908222554734](https://i.loli.net/2021/09/08/XYvfj4OTt89rdhq.png)



<video src=".\Vedio\SAO漫游.mp4"></video>

![image-20210908223236798](https://i.loli.net/2021/09/08/Sja6m38PRiTFL4r.png)

![image-20210908223326871](https://i.loli.net/2021/09/08/SLT8p5bo6yfWNU9.png)

![image-20210908223351612](https://i.loli.net/2021/09/08/TfHQGXuxKACEJir.png)

![image-20210908223533551](https://i.loli.net/2021/09/08/Q1T9LqdKPeWClaX.png)

![image-20210908223557372](https://i.loli.net/2021/09/08/HYvCpPrSOuGBUjQ.png)

![image-20210908223426192](https://i.loli.net/2021/09/08/lUAfa4W3e1vXV6O.png)

### Multi Scale Volumetric Obscurance

Thickness Modifier：修改遮挡器的厚度。这会增加黑暗区域，但会在物体周围引入暗晕

![image-20210908224722563](https://i.loli.net/2021/09/08/yEtVLrovs1wazRg.png)

![image-20210908224803882](https://i.loli.net/2021/09/08/Rm7Sil5HohGIayA.png)

直观上感觉MSVO比SAO计算量大，因为在运动中有抖动现象，同时感觉MSVO比SAO过渡更柔和

<video src=".\Vedio\SAO漫游.mp4"></video>

<video src=".\Vedio\MSVO漫游2.mp4"></video>

![image-20210908225505013](https://i.loli.net/2021/09/08/qnDfpEOLzlxK8a9.png)

![image-20210908225408497](https://i.loli.net/2021/09/08/ayHVoxfYBrGDjFA.png)

![image-20210908225436922](https://i.loli.net/2021/09/08/hMIJXQ2b4qmV7LS.png)

![image-20210908225708345](https://i.loli.net/2021/09/08/Nfk6n8ryDQw5Xv9.png)

![image-20210908225641016](https://i.loli.net/2021/09/08/KMBzEgQ2OFdP6Ww.png)

![image-20210908225734286](https://i.loli.net/2021/09/08/lnpDKM3VSwifNz9.png)

## 其他AO方案

### HBAO

code refer:[UnityCrytekSponza-master2019/Hbao.shader at master · lync-li/UnityCrytekSponza-master2019 (github.com)](https://github.com/lync-li/UnityCrytekSponza-master2019/blob/master/Assets/Render/Shader/PostProcessShader/Hbao.shader)

PPT:https://developer.download.nvidia.cn/presentations/2008/SIGGRAPH/HBAO_SIG08b.pdf

学习链接：[HBAO(屏幕空间的环境光遮蔽) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/103683536)

HBAO（Image-space horizon-based ambient occlusion）是SSAO的改进，通过以下步骤进行AO：

1.对屏幕上一点像素P，分四方向进行光线步进（Ray Marching），每一个像素不能用相同的四个方向，所以要在方向上旋转一个随机角度

2.对其任意一方向得到一维高度场，在此方向步进（Marching）得到最大水平角（Horizon angle）

3.根据点P和法线N求切面角（tangent angle）

4.通过2,3得到的水平角和切面角，利用计算公式【AO(θ)=sin horizon(θ)-sin tangent(θ)】求出AO值

5.对剩下3方向做同样操作，得到点P四方形上(2D)AO值,相加后取平均得最终AO值

#### HBAO问题：

1.带状不连续问题（在SAO部分出现的那个问题），通过给切线角增加一个偏移量可以解决

![image-20210909115253617](https://i.loli.net/2021/09/09/yxej9BL3g8HtvOz.png)

![image-20210909115323837](https://i.loli.net/2021/09/09/rEHqJeQtUhkbG6d.png)![image-20210909115324069](https://i.loli.net/2021/09/09/rEHqJeQtUhkbG6d.png)

2.ao值相差过大问题，在SSAO中也遇到过，直接输出AO值会导致周围几个AO值差过大，对比度太强，阴影过硬，可以用以下衰减公式解决

```c
ao = saturate(1.0 - ao);//衰减,解决AO差值过大的不连续问题
```

3.随机噪点，可以用模糊解决，同理SSAO

不模糊噪点时分明显

![image-20210909120301899](https://i.loli.net/2021/09/09/cD1fz3dMAIQH26N.png)

模糊后：

![image-20210909120353521](https://i.loli.net/2021/09/09/b84UMwQxGVHSvrf.png)

#### 核心代码

```c
half4 frag_ao(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target {	
	//float3 P = FetchViewPos(i.uv2,depthNormal.zw);
	float3 P = FetchViewPos(i.uv2);//当前点

	clip(_MaxDistance - P.z);
	
	float stepSize = min((_AORadius / P.z), 128) / (STEPS + 1.0);//光线步进步长

	// (cos(alpha), sin(alpha), jitter)
	float3 rand = tex2D(_NoiseTex, screenPos.xy / 4.0).rgb;

	float2 InvScreenParams = _ScreenParams.zw - 1.0;
	
#if NORMALS_RECONSTRUCT
		float3 Pr, Pl, Pt, Pb;
		Pr = FetchViewPos(i.uv2 + float2(InvScreenParams.x, 0));
		Pl = FetchViewPos(i.uv2 + float2(-InvScreenParams.x, 0));
		Pt = FetchViewPos(i.uv2 + float2(0, InvScreenParams.y));
		Pb = FetchViewPos(i.uv2 + float2(0, -InvScreenParams.y));
		float3 N = normalize(cross(MinDiff(P, Pr, Pl), MinDiff(P, Pt, Pb)));
#else
	float4 normal = tex2D(_NormalBufferTex, i.uv2);
	float3 N = DecodeNormal(normal.xy);
	//float3 N = mul((float3x3)(UNITY_MATRIX_I_V),viewNormal);	
	//float3 N = normal * 2 - 1;
#endif	

	N = float3(N.x, -N.yz);

	float ao = 0;

	UNITY_UNROLL
	for (int d = 0; d < DIRECTIONS; ++d) {//对各方向
		float2 direction = RotateDirections(Directions[d], rand.xy);//随机旋转角度

		// Jitter starting sample within the first step
		float rayPixels = (rand.z * stepSize + 1.0);

		UNITY_UNROLL
		for (int s = 0; s < STEPS; ++s) {//进行光线步进

			float2 snappedUV = round(rayPixels * direction) * InvScreenParams  + i.uv2;
			
			//float4 depthNormal2 = tex2D(_CameraDepthNormalsTexture, snappedUV);
			
			//float3 S = FetchViewPos(snappedUV,depthNormal2.zw);
			float3 S = FetchViewPos(snappedUV);

			rayPixels += stepSize;

			float contrib = ComputeAO(P, N, S);
			ao += contrib;

		}
	}
		
	ao *= (_AOmultiplier / (STEPS * DIRECTIONS));

	ao = saturate(1.0 - ao);//衰减公式,解决AO差值过大的不连续问题
	
	return half4(ao, 0,0,1);
}
```

```c
//分四个方向步进
#define DIRECTIONS		4
//光线步进次数
#define STEPS			3
static const float2 Directions[8] = {
    float2(1,0),
    float2(-0.5000001,0.8660254),
    float2(-0.4999999,-0.8660254),
	float2(0,1),	
};

half3 DecodeNormal (half2 enc)
{
    half2 fenc = enc*4-2;
    half f = dot(fenc,fenc);
    half g = sqrt(1-f/4);
    half3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}


//inline float3 FetchViewPos(float2 uv,float2 depth) {	
//	float z = DecodeFloatRG (depth) * _ProjectionParams.z;
//	return float3((uv * _UVToView.xy + _UVToView.zw) * z, z);
//}

inline float3 FetchViewPos(float2 uv) {	
	float z = DECODE_EYEDEPTH(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv));
	return float3((uv * _UVToView.xy + _UVToView.zw) * z, z);
}


inline float Falloff(float distanceSquare) {
	// 1 scalar mad instruction
	return distanceSquare * _NegInvRadius2 + 1.0;
}

inline float ComputeAO(float3 P, float3 N, float3 S) {
	float3 V = S - P;
	float VdotV = dot(V, V);
	VdotV = clamp(VdotV,0.01,10000); // fixed precision
	float NdotV = dot(N, V) * rsqrt(VdotV);

	//return saturate(Falloff(VdotV));
	// Use saturate(x) instead of max(x,0.f) because that is faster on Kepler
	return saturate(NdotV - _AngleBias) * saturate(Falloff(VdotV));
}

inline float3 MinDiff(float3 P, float3 Pr, float3 Pl) {
	float3 V1 = Pr - P;
	float3 V2 = P - Pl;
	return (dot(V1, V1) < dot(V2, V2)) ? V1 : V2;
}

inline float2 RotateDirections(float2 dir, float2 rot) {
	return float2(dir.x * rot.x - dir.y * rot.y,
					dir.x * rot.y + dir.y * rot.x);
}
```

#### 结果

![image-20210909164730092](https://i.loli.net/2021/09/09/Peps57iWxvlRfS2.png)

<video src=".\Vedio\HBAO漫游.mp4"></video>

![image-20210909165218778](https://i.loli.net/2021/09/09/wQhgSoCrMZBcqsL.png)

![image-20210909165404135](https://i.loli.net/2021/09/09/iFcxA795mQvt6Vs.png)

![image-20210909165438982](https://i.loli.net/2021/09/09/swGqQKJBl6bvdXU.png)

![image-20210909165724541](https://i.loli.net/2021/09/09/kFnI9SMQo2LP1G6.png)

![image-20210909165642972](https://i.loli.net/2021/09/09/XpaUQdqnJPrRLN5.png)

![image-20210909165619100](https://i.loli.net/2021/09/09/TdM3lq4XJgseOrF.png)

### GTAO

code refer:[MaxwellGengYF/Unity-Ground-Truth-Ambient-Occlusion: A physically based screen space ambient occulsion post processing effect (github.com)](https://github.com/MaxwellGengYF/Unity-Ground-Truth-Ambient-Occlusion)

PPT：https://iryoku.com/downloads/Practical-Realtime-Strategies-for-Accurate-Indirect-Occlusion.pdf

学习链接：[UE4 Mobile GTAO 实现(HBAO续) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/145339736)

[Unity_GroundTruth-Oclusion - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/53097274)

[实时渲染器开发(四) GTAO原理与实践 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/342210503)

GTAO（**GroundTruth-Oclusion**）在HBAO基础上增加余弦项Cosine Weight操作使AO计算的物理性更精准，同时还增加了AO的MutiBounce弹射效果。另外，在AO基础上增加了RO（ReflectionOcclusion）。可以利用双边滤波，Temporal滤波来优化最终AO。

#### 部分核心步骤

##### Cosine Weight



```c
half IntegrateArc_CosWeight(half2 h, half n)//计算余弦项
{
    half2 Arc = -cos(2 * h - n) + cos(n) + 2 * h * sin(n);
    return 0.25 * (Arc.x + Arc.y);
}
```

##### Bent Normal



```c
//利用h1和h2和viewDir构建一个坐标系去推算出BentAngle以重建BentNormal
bentAngle = (h.x + h.y) * 0.5;//average both horizon angles
BentNormal += viewDir * cos(bentAngle) - tangent * sin(bentAngle);
```

##### Multi-Bounce



```c
inline half3 MultiBounce(half AO, half3 Albedo)
{
	half3 A = 2 * Albedo - 0.33;
	half3 B = -4.8 * Albedo + 0.64;
	half3 C = 2.75 * Albedo + 0.69;
	return max(AO, ((AO * A + B) * AO + C) * AO);
}
```

##### RO



```c
//计算RO
inline half ReflectionOcclusion(half3 BentNormal, half3 ReflectionVector, half Roughness, half OcclusionStrength)
{
	half BentNormalLength = length(BentNormal);
	half ReflectionConeAngle = max(Roughness, 0.1) * PI;
	half UnoccludedAngle = BentNormalLength * PI * OcclusionStrength;

	half AngleBetween = acos(dot(BentNormal, ReflectionVector) / max(BentNormalLength, 0.001));
	half ReflectionOcclusion = ApproximateConeConeIntersection(ReflectionConeAngle, UnoccludedAngle, AngleBetween);//BentNormal作为VisibilityCone去和SpecularCone求交得到ReflectionOcclusion
	ReflectionOcclusion = lerp(0, ReflectionOcclusion, saturate((UnoccludedAngle - 0.1) / 0.2));
	return ReflectionOcclusion;
}

inline half ReflectionOcclusion_Approch(half NoV, half Roughness, half AO)
{
	return saturate(pow(NoV + AO, Roughness * Roughness) - 1 + AO);
}
```

##### AO



```c
for (int i = 0; i < NumCircle; i++)
{
    angle = (i + noiseDirection + _AO_TemporalDirections) * (UNITY_PI / (half)NumCircle);
    sliceDir = half3(half2(cos(angle), sin(angle)), 0);
    slideDir_TexelSize = sliceDir.xy * _AO_RT_TexelSize.xy;
    h = -1;

    for (int j = 0; j < NumSlice; j++)
    {
        uvOffset = slideDir_TexelSize * max(stepRadius * (j + initialRayStep), 1 + j);
        uvSlice = uv.xyxy + float4(uvOffset.xy, -uvOffset);

        ds = GetPosition(uvSlice.xy) - vPos;
        dt = GetPosition(uvSlice.zw) - vPos;

        dsdt = half2(dot(ds, ds), dot(dt, dt));
        dsdtLength = rsqrt(dsdt);

        falloff = saturate(dsdt.xy * (2 / pow2(radius)));

        H = half2(dot(ds, viewDir), dot(dt, viewDir)) * dsdtLength;
        h.xy = (H.xy > h.xy) ? lerp(H, h, falloff) : lerp(H.xy, h.xy, thickness);
    }

    planeNormal = normalize(cross(sliceDir, viewDir));
    tangent = cross(viewDir, planeNormal);
    projectedNormal = viewNormal - planeNormal * dot(viewNormal, planeNormal);
    projLength = length(projectedNormal);

    cos_n = clamp(dot(normalize(projectedNormal), viewDir), -1, 1);
    n = -sign(dot(projectedNormal, tangent)) * acos(cos_n);

    h = acos(clamp(h, -1, 1));
    h.x = n + max(-h.x - n, -UNITY_HALF_PI);
    h.y = n + min(h.y - n, UNITY_HALF_PI);

    //利用h1和h2和viewDir构建一个坐标系去推算出BentAngle以重建BentNormal
    bentAngle = (h.x + h.y) * 0.5;//average both horizon angles

    BentNormal += viewDir * cos(bentAngle) - tangent * sin(bentAngle);
    Occlusion += projLength * IntegrateArc_CosWeight(h, n);	
}
```

### 烘焙lightmap

可生成静态物体见的AO信息，通过Generate Lightmap UVs可以生成第二个纹理坐标数据以存储贴图信息

当然代价是游戏场景要储存额外的贴图信息，以空间换取时间效率

![image-20210904205423265](https://i.loli.net/2021/09/04/kBivugwIZrKcT87.png)

![image-20210904205010036](https://i.loli.net/2021/09/04/ANf1cP5rZGJUejh.png)

## 总结

实时的AO物理的正确性较差，但对采样方式以及对周围采样区域的计算方式的优化可以提升直观的视觉体验或者生成AO的速度，相对于静态烘焙的AO，实时AO有了更好的交互，但是随着人物和周围建筑距离的变化，AO的误差也会增加，造成AO区域的抖动或整个场景过暗情况。

## 备注

屏幕空间的算法依赖于深度信息，这里汇总一些对学习求取深度信息的链接

[tex2Dproj和tex2D的区别 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/107627483)

[(11条消息) Unity Shader计算深度并显示_TDC的专栏-CSDN博客](https://blog.csdn.net/ak47007tiger/article/details/102657908?utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.control)

[Unity从深度缓冲重建世界空间位置 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/92315967)