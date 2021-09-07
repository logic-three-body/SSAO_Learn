using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HABOEFECT : MonoBehaviour
{
    public Shader _shader=null;
    private Material hbaoMaterial;
    private Camera cam;
    [Range(0f,1f)]
    public float aoStrength = 0f; 
    [Range(4, 64)]
    public int SampleKernelCount = 64;
    private List<Vector4> sampleKernelList = new List<Vector4>();
    [Range(0.0001f,10f)]
    public float sampleKeneralRadius = 0.01f;
    
    [Range(0.0001f,1f)]
    public float rangeStrength = 0.001f;
    
    public float depthBiasValue;

    public Texture Nosie;//噪声贴图

    [Range(0, 2)]
    public int DownSample = 0;

    [Range(1, 4)]
    public int BlurRadius = 2;
    [Range(0, 0.2f)]
    public float bilaterFilterStrength = 0.2f;
   // public bool OnlyShowAO = false;

    public Vector3 RandomVec;//自己调RandomVector debug用

    [Range(0, 1)]
    public int isRandom = 0;//决定是自己调RandomVector还是采用随机RandomVector

    //HBAO
    public float _MaxPixelRadius;
    public float _RayMarchingStep;
    public float _RayAngleStep;
    public float _AngleBiasValue;
    public float _AORadius;

    public enum SSAOPassName
    {
        GenerateAO = 0,//pass0
        BilateralFilter = 1,//pass1
        Composite = 2,//pass2
    }

    public enum DebugMode
    {
        ONLY_AO=0,//仅输出AO
        BLUR_AO=1,//输出模糊后的AO
        COMPLETE=2,//完全输出
        NO_BLUR=3//完全输出但不模糊AO
    }

    public DebugMode mode;

    private void Awake()
    {
        var shader = Shader.Find("ImageEffect/HBAO");
        shader = _shader;
        hbaoMaterial = new Material(shader);
    }

    //获取深度&法线缓存数据
    private void Start()
    {
        cam = this.GetComponent<Camera>();
        //相机渲染模式为带深度和法线
        cam.depthTextureMode = cam.depthTextureMode | DepthTextureMode.DepthNormals;
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        GenerateAOSampleKernel();
        int rtW = source.width >> DownSample;
        int rtH = source.height >> DownSample;

        //AO
        RenderTexture aoRT = RenderTexture.GetTemporary(rtW,rtH,0);
        hbaoMaterial.SetVectorArray("_SampleKernelArray", sampleKernelList.ToArray());
        hbaoMaterial.SetFloat("_RangeStrength", rangeStrength);
        hbaoMaterial.SetFloat("_AOStrength", aoStrength);
        hbaoMaterial.SetFloat("_SampleKernelCount", sampleKernelList.Count);
        hbaoMaterial.SetFloat("_SampleKeneralRadius",sampleKeneralRadius);
        hbaoMaterial.SetFloat("_DepthBiasValue",depthBiasValue);
        hbaoMaterial.SetFloat("_MaxPixelRadius", _MaxPixelRadius);
        hbaoMaterial.SetFloat("_RayAngleStep", _RayAngleStep);
        hbaoMaterial.SetFloat("_AngleBiasValue", _AngleBiasValue);
        hbaoMaterial.SetFloat("_AORadius", _AORadius);

        hbaoMaterial.SetTexture("_NoiseTex", Nosie);
        hbaoMaterial.SetVector("_randomVec", RandomVec);//传入随机向量      
        hbaoMaterial.SetInt("_isRandom", isRandom);//决定使用随机分布产生随机向量还是使用自己调节的固定随机向量
        Graphics.Blit(source, aoRT, hbaoMaterial,(int)SSAOPassName.GenerateAO);
        //Blur
        RenderTexture blurRT = RenderTexture.GetTemporary(rtW,rtH,0);
        hbaoMaterial.SetFloat("_BilaterFilterFactor", 1.0f - bilaterFilterStrength);
        hbaoMaterial.SetVector("_BlurRadius", new Vector4(BlurRadius, 0, 0, 0));
        Graphics.Blit(aoRT, blurRT, hbaoMaterial, (int)SSAOPassName.BilateralFilter);

        hbaoMaterial.SetVector("_BlurRadius", new Vector4(0, BlurRadius, 0, 0));
        if (DebugMode.ONLY_AO==mode)
        {
            Graphics.Blit(aoRT, destination);
        }
        else if(DebugMode.BLUR_AO == mode)
        {
            Graphics.Blit(blurRT, destination, hbaoMaterial, (int)SSAOPassName.BilateralFilter);
        }
        else if(DebugMode.COMPLETE == mode)
        {
            Graphics.Blit(blurRT, aoRT, hbaoMaterial, (int)SSAOPassName.BilateralFilter);
            hbaoMaterial.SetTexture("_AOTex", aoRT);
            Graphics.Blit(source, destination, hbaoMaterial, (int)SSAOPassName.Composite);
        }
        else
        {
            hbaoMaterial.SetTexture("_AOTex", aoRT);
            Graphics.Blit(source, destination, hbaoMaterial, (int)SSAOPassName.Composite);
        }

        RenderTexture.ReleaseTemporary(aoRT);
        RenderTexture.ReleaseTemporary(blurRT);
    }

    //refer:https://blog.csdn.net/qq_39300235/article/details/102460405
    private void GenerateAOSampleKernel()
    {
        if (SampleKernelCount == sampleKernelList.Count)
            return;
        sampleKernelList.Clear();
        for (int i = 0; i < SampleKernelCount; i++)//在此生成随机样本
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
}
