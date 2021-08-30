//https://blog.csdn.net/tianhai110/article/details/5684128?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-7.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-7.channel_param
Shader "ImageEffect/SSAO"
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
		// NDC position
		float4 ndcPos = (screenPos / screenPos.w) * 2 - 1;
		// 计算至远屏幕方向
		float3 clipVec = float3(ndcPos.x, ndcPos.y, 1.0) * _ProjectionParams.z;
		o.viewVec = mul(unity_CameraInvProjection, clipVec.xyzz).xyz;
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

		//获取像素相机屏幕法线，法相z方向相对于相机为负（so 需要乘以-1置反），并处理成单位向量
		viewNormal = normalize(viewNormal) * float3(1, 1, -1);

		//铺平纹理
		float2 noiseScale = _ScreenParams.xy / 4.0;
		float2 noiseUV = i.uv * noiseScale;
		//randvec法线半球的随机向量
		float3 randvec = tex2D(_NoiseTex,noiseUV).xyz;
		//Gramm-Schimidt处理创建正交基
		//法线&切线&副切线构成的坐标空间
		float3 tangent = normalize(randvec - viewNormal * dot(randvec,viewNormal));
		float3 bitangent = cross(viewNormal,tangent);
		float3x3 TBN = float3x3(tangent,bitangent,viewNormal);

		//采样核心
		float ao = 0;
		int sampleCount = _SampleKernelCount;//每个像素点上的采样次数
		//https://blog.csdn.net/qq_39300235/article/details/102460405
		for(int i=0;i<sampleCount;i++){
			//随机向量，转化至法线切线空间中
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
			DecodeDepthNormal(rcdn, randomDepth, randomNormal);
			
			//判断累加ao值
			float range = abs(randomDepth - linear01Depth) > _RangeStrength ? 0.0 : 1.0;
			float selfCheck = randomDepth + _DepthBiasValue < linear01Depth ? 1.0 : 0.0;

			//采样点的深度值和样本深度比对前后关系
			ao += range * selfCheck * weight;
		}

		ao = ao/sampleCount;
		ao = max(0.0, 1 - ao * _AOStrength);
		return float4(ao,ao,ao,1);
    }
	
	//Blur
	float _BilaterFilterFactor;
	float2 _MainTex_TexelSize;
	float2 _BlurRadius;

	///基于法线的双边滤波（Bilateral Filter）
	//https://blog.csdn.net/puppet_master/article/details/83066572
	float3 GetNormal(float2 uv)
	{
		float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);	
		return DecodeViewNormalStereo(cdn);
	}

	half CompareNormal(float3 nor1,float3 nor2)
	{
		return smoothstep(_BilaterFilterFactor,1.0,dot(nor1,nor2));
	}
	
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
		//Pass 1 : Bilateral Filter Blur
		Pass
		{
			CGPROGRAM
			#pragma vertex vert_Ao
			#pragma fragment frag_Blur
			ENDCG
		}

		//Pass 2 : Composite AO
		Pass
		{
			CGPROGRAM
			#pragma vertex vert_Ao
			#pragma fragment frag_Composite
			ENDCG
		}
    }
}
