#include "UnityCG.cginc"
inline float2 RotateDirections(float2 dir, float2 rot) {
	return float2(dir.x * rot.x - dir.y * rot.y,
				  dir.x * rot.y + dir.y * rot.x);
}
 
inline float Falloff2(float distance, float radius)
{
	float a = distance / radius;
	return clamp(1.0 - a * a, 0.0, 1.0);
}
 

 
// Reconstruct view-space position from UV and depth.
// p11_22 = (unity_CameraProjection._11, unity_CameraProjection._22)
// p13_31 = (unity_CameraProjection._13, unity_CameraProjection._23)
float3 ReconstructViewPos(float2 uv)
{
	float3x3 proj = (float3x3)unity_CameraProjection;
	float2 p11_22 = float2(unity_CameraProjection._11, unity_CameraProjection._22);
	float2 p13_31 = float2(unity_CameraProjection._13, unity_CameraProjection._23);
	float depth;
	float3 viewNormal;
	float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
	DecodeDepthNormal(cdn, depth, viewNormal);
	depth *= _ProjectionParams.z;
	return float3((uv * 2.0 - 1.0 - p13_31) / p11_22 * (depth), depth);
}
 
inline float2 GetRayMarchingDir(float angle)
{
	float sinValue, cosValue;
	sincos(angle, sinValue, cosValue);
	return RotateDirections(float2(cosValue, sinValue), float2(1.0, 0));
}
 
