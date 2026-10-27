#version 330 core
layout(location = 0) out float FilteredDepth;

in vec2 TexCoords;

uniform sampler2D colorTex;
uniform sampler2D depthTex;

uniform float width;
uniform float height;
uniform float isYCbCr; // 判斷是否為 YUV 格式
uniform float chroma_offset;

// 濾波器參數
uniform float sigmaSpatial;
uniform float sigmaColor;
uniform float edgeThreshold; // 建議從 C++ 傳入 0.08 ~ 0.12

// 提取單點色彩引導值 (標準 RGB Luma 權重)
float getGuideValue(vec2 uv) {
    if(isYCbCr > 0.5f){
        vec2 texcoord_Y = vec2(uv.x, uv.y * height / (height * 1.5f + chroma_offset));
        return texture(colorTex, texcoord_Y).r;
    } else {
        vec3 c = texture(colorTex, uv).rgb;
        return dot(c, vec3(0.299f, 0.587f, 0.114f));
    }
}

// 計算彩色圖在當前 UV 點的 Sobel 梯度強度
float getGuideGradient(vec2 uv) {
    vec2 texel = vec2(1.0f / width, 1.0f / height);
    
    // 讀取周圍 3x3 點的引導值
    float tL = getGuideValue(uv + vec2(-texel.x,  texel.y));
    float tM = getGuideValue(uv + vec2( 0.0f,     texel.y));
    float tR = getGuideValue(uv + vec2( texel.x,  texel.y));
    float mL = getGuideValue(uv + vec2(-texel.x,  0.0f));
    float mR = getGuideValue(uv + vec2( texel.x,  0.0f));
    float bL = getGuideValue(uv + vec2(-texel.x, -texel.y));
    float bM = getGuideValue(uv + vec2( 0.0f,    -texel.y));
    float bR = getGuideValue(uv + vec2( texel.x, -texel.y));

    // Sobel 矩陣算子
    float gradX = (tR + 2.0f * mR + bR) - (tL + 2.0f * mL + bL);
    float gradY = (bL + 2.0f * bM + bR) - (tL + 2.0f * tM + tR);

    return sqrt(gradX * gradX + gradY * gradY);
}

void main()
{
    vec2 tex_offset = vec2(1.0f / width, 1.0f / height);
    
    // 1. 取得中心點的深度與色彩引導值
    float depthCenter = texture(depthTex, TexCoords).r;
    float guideCenter = getGuideValue(TexCoords);

    // 2. 計算中心點的色彩梯度強度
    float gradCenter = getGuideGradient(TexCoords);

    // 3. 【梯度自適應邏輯優化】：
    // 當 gradCenter 低於 edgeThreshold 時（代表是花紋或平坦區），
    // 使用 smoothstep 平滑放寬 sigmaColor（最多放大 2.0 倍），讓深度機能穿透花紋進行平滑；
    // 當 gradCenter 高於 edgeThreshold 時（強輪廓邊界），保持 1.0 倍 sigmaColor 嚴格鎖邊。
    float factor = smoothstep(0.0f, edgeThreshold, gradCenter);
    float adaptiveSigmaColor = mix(sigmaColor * 2.0f, sigmaColor, factor);

    // 4. 預先計算高斯分母
    float twoSigmaSpatialSq = 2.0f * sigmaSpatial * sigmaSpatial;
    float twoSigmaColorSq   = 2.0f * adaptiveSigmaColor * adaptiveSigmaColor;

    float weightSum = 0.0f;
    float depthSum = 0.0f;
    int kernelRadius = 4; // 9x9 視窗，兼顧速度與範圍

    // 5. 雙迴圈進行雙邊加權
    for(int y = -kernelRadius; y <= kernelRadius; y++) {
        for(int x = -kernelRadius; x <= kernelRadius; x++) {
            vec2 offset = vec2(float(x), float(y)) * tex_offset;
            vec2 sampleUV = clamp(TexCoords + offset, vec2(0.0f), vec2(1.0f));

            float sampleDepth = texture(depthTex, sampleUV).r;
            float sampleGuide = getGuideValue(sampleUV);
            
            // 空間權重
            float dist2 = float(x*x + y*y);
            float spatialWeight = exp(-dist2 / twoSigmaSpatialSq);
            
            // 範圍權重（使用自適應 sigmaColor）
            float colorDist = sampleGuide - guideCenter;
            float colorWeight = exp(-(colorDist * colorDist) / twoSigmaColorSq);
            
            float w = spatialWeight * colorWeight;
            
            depthSum += sampleDepth * w;
            weightSum += w;
        }
    }
    
    FilteredDepth = (weightSum > 0.00001f) ? (depthSum / weightSum) : depthCenter;
}