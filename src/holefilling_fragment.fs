#version 430 core
layout(location = 0) out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D warpedColorTex;
uniform sampler2D warpedDepthTex;

layout(std430, binding = 2) buffer OtsuResultBuffer {
    float u_OtsuThreshold;
};

uniform float width;
uniform float height;
uniform float sigma_c = 0.5;
uniform float sigma_d = 50.0;
uniform float sigma_s = 30.0;
uniform float zNear;
uniform float zFar;

const float HOLE_DEPTH = 9999.0;
const float ALPHA_THRESHOLD = 0.1;

const vec2 directions[16] = vec2[](
    vec2(1.0, 0.0),   vec2(0.923, 0.382),
    vec2(0.707, 0.707), vec2(0.382, 0.923),
    vec2(0.0, 1.0),   vec2(-0.382, 0.923),
    vec2(-0.707, 0.707), vec2(-0.923, 0.382),
    vec2(-1.0, 0.0),  vec2(-0.923, -0.382),
    vec2(-0.707, -0.707), vec2(-0.382, -0.923),
    vec2(0.0, -1.0),  vec2(0.382, -0.923),
    vec2(0.707, -0.707), vec2(0.923, -0.382)
);

bool insideImage(vec2 uv)
{
    return uv.x >= 0.0 && uv.x < 1.0 &&
           uv.y >= 0.0 && uv.y < 1.0;
}

bool validPixel(vec2 uv)
{
    if (!insideImage(uv))
        return false;

    // 從 .g 通道讀取真正的深度值
    float depth = texture(warpedDepthTex, uv).g;
    float alpha = texture(warpedColorTex, uv).a;

    return depth < HOLE_DEPTH && alpha > ALPHA_THRESHOLD;
}

void main()
{
    vec4 currentColor = texture(warpedColorTex, TexCoords);
    float currentDepth = texture(warpedDepthTex, TexCoords).g;

    // 原始像素有效時直接輸出
    if (currentDepth < HOLE_DEPTH && currentColor.a > ALPHA_THRESHOLD)
    {
        FragColor = currentColor;
        return;
    }

    vec2 texel = vec2(1.0 / width, 1.0 / height);

    // ====================================================================
    // 【動態 Otsu 門檻】：將 Otsu 計算出來的 0~1 比例還原成 FBO 物理深度
    // ====================================================================
    float normOtsu = clamp(u_OtsuThreshold, 0.0, 1.0);
    float realDepthThreshold = normOtsu * (zFar - zNear) + zNear;

    vec4 finalColorSum = vec4(0.0);
    float finalWeightSum = 0.0;

    const int maxSearch = 200;

    vec4 fallbackColor = vec4(0.0);
    float minDistance = 1e20;
    bool foundAnyPixel = false;

    for (int k = 0; k < 16; ++k)
    {
        vec2 dir = directions[k] * texel;
        vec2 q_k_uv = vec2(-1.0);
        float d_k = 0.0;

        for (int t = 1; t <= maxSearch; ++t)
        {
            vec2 sampleUV = TexCoords + dir * float(t);

            if (!insideImage(sampleUV))
                break;

            if (validPixel(sampleUV))
            {
                float depth_sample = texture(warpedDepthTex, sampleUV).g;

                // 直接使用物理深度進行比較，若是近處前景則立即捨棄此方向
                if (depth_sample < realDepthThreshold)
                    break; 

                q_k_uv = sampleUV;
                d_k = float(t);

                if (d_k < minDistance)
                {
                    minDistance = d_k;
                    fallbackColor = texture(warpedColorTex, sampleUV);
                    foundAnyPixel = true;
                }
                break;
            }
        }

        if (q_k_uv.x < 0.0)
            continue;

        float depth_qk = texture(warpedDepthTex, q_k_uv).g;
        vec4 color_qk = texture(warpedColorTex, q_k_uv);

        float S_k_sum = 0.0;
        int n_k = 0;

        int stepOffset = int(d_k) + 1;

        for (int j = 0; j < 4 && stepOffset <= maxSearch; ++stepOffset)
        {
            vec2 r_uv = TexCoords + dir * float(stepOffset);

            if (!insideImage(r_uv))
                break;

            if (!validPixel(r_uv))
                continue;

            float depth_r = texture(warpedDepthTex, r_uv).g;
            
            // Consistency 驗證階段也必須防止採樣到前景
            if (depth_r < realDepthThreshold)
                break;

            vec4 color_r = texture(warpedColorTex, r_uv);

            vec3 colorDiff = color_qk.rgb - color_r.rgb;
            float colorDist2 = dot(colorDiff, colorDiff);

            float depthDiff = depth_qk - depth_r;
            float depthDist2 = depthDiff * depthDiff;

            float sigmaC2 = max(sigma_c * sigma_c, 1e-8);
            float sigmaD2 = max(sigma_d * sigma_d, 1e-8);

            float w_c = exp(-colorDist2 / (2.0 * sigmaC2));
            float w_d = exp(-depthDist2 / (2.0 * sigmaD2));

            S_k_sum += w_c * w_d;
            ++n_k;
            ++j;
        }

        if (n_k == 0)
            continue;

        float S_k = S_k_sum / float(n_k);

        float sigmaS2 = max(sigma_s * sigma_s, 1e-8);
        float w_s = exp(-(d_k * d_k) / (2.0 * sigmaS2));

        float w_k = S_k * w_s;

        finalColorSum += color_qk * w_k;
        finalWeightSum += w_k;
    }

    if (finalWeightSum > 1e-8)
    {
        FragColor = finalColorSum / finalWeightSum;
    }
    else if (foundAnyPixel)
    {
        FragColor = fallbackColor;
    }
    else
    {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}