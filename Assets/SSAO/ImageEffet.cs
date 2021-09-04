using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImageEffet : MonoBehaviour
{
    public Shader _shader=null;
    private Material ssaoMaterial;
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
        var shader = Shader.Find("ImageEffect/SSAO");
        shader = _shader;
        ssaoMaterial = new Material(shader);
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
        ssaoMaterial.SetVectorArray("_SampleKernelArray", sampleKernelList.ToArray());
        ssaoMaterial.SetFloat("_RangeStrength", rangeStrength);
        ssaoMaterial.SetFloat("_AOStrength", aoStrength);
        ssaoMaterial.SetFloat("_SampleKernelCount", sampleKernelList.Count);
        ssaoMaterial.SetFloat("_SampleKeneralRadius",sampleKeneralRadius);
        ssaoMaterial.SetFloat("_DepthBiasValue",depthBiasValue);
        ssaoMaterial.SetTexture("_NoiseTex", Nosie);
        ssaoMaterial.SetVector("_randomVec", RandomVec);//传入随机向量      
        ssaoMaterial.SetInt("_isRandom", isRandom);//决定使用随机分布产生随机向量还是使用自己调节的固定随机向量
        Graphics.Blit(source, aoRT, ssaoMaterial,(int)SSAOPassName.GenerateAO);
        //Blur
        RenderTexture blurRT = RenderTexture.GetTemporary(rtW,rtH,0);
        ssaoMaterial.SetFloat("_BilaterFilterFactor", 1.0f - bilaterFilterStrength);
        ssaoMaterial.SetVector("_BlurRadius", new Vector4(BlurRadius, 0, 0, 0));
        Graphics.Blit(aoRT, blurRT, ssaoMaterial, (int)SSAOPassName.BilateralFilter);

        ssaoMaterial.SetVector("_BlurRadius", new Vector4(0, BlurRadius, 0, 0));
        if (DebugMode.ONLY_AO==mode)
        {
            Graphics.Blit(aoRT, destination);
        }
        else if(DebugMode.BLUR_AO == mode)
        {
            Graphics.Blit(blurRT, destination, ssaoMaterial, (int)SSAOPassName.BilateralFilter);
        }
        else if(DebugMode.COMPLETE == mode)
        {
            Graphics.Blit(blurRT, aoRT, ssaoMaterial, (int)SSAOPassName.BilateralFilter);
            ssaoMaterial.SetTexture("_AOTex", aoRT);
            Graphics.Blit(source, destination, ssaoMaterial, (int)SSAOPassName.Composite);
        }
        else
        {
            ssaoMaterial.SetTexture("_AOTex", aoRT);
            Graphics.Blit(source, destination, ssaoMaterial, (int)SSAOPassName.Composite);
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
