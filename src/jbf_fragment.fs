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
uniform float edgeThreshold; // 新增：邊緣門檻 (預設可設 0.15)

// 提取單點色彩引導值 (Luma/Red)
float getGuideValue(vec2 uv) {
    if(isYCbCr > 0.5f){
        vec2 texcoord_Y = vec2(uv.x, uv.y * height / (height * 1.5f + chroma_offset));
        return texture(colorTex, texcoord_Y).r;
    } else {
        return texture(colorTex, uv).r;
    }
}

// 計算彩色圖在當前 UV 點的 Sobel 梯度強度
float getGuideGradient(vec2 uv) {
    vec2 texel = vec2(1.0 / width, 1.0 / height);
    
    // 讀取周圍 3x3 點的引導值
    float tL = getGuideValue(uv + vec2(-texel.x,  texel.y));
    float tM = getGuideValue(uv + vec2( 0.0,      texel.y));
    float tR = getGuideValue(uv + vec2( texel.x,  texel.y));
    float mL = getGuideValue(uv + vec2(-texel.x,  0.0));
    float mR = getGuideValue(uv + vec2( texel.x,  0.0));
    float bL = getGuideValue(uv + vec2(-texel.x, -texel.y));
    float bM = getGuideValue(uv + vec2( 0.0,     -texel.y));
    float bR = getGuideValue(uv + vec2( texel.x, -texel.y));

    // Sobel 矩陣算子
    float gradX = (tR + 2.0 * mR + bR) - (tL + 2.0 * mL + bL);
    float gradY = (bL + 2.0 * bM + bR) - (tL + 2.0 * tM + tR);

    return sqrt(gradX * gradX + gradY * gradY);
}

void main()
{
    vec2 tex_offset = vec2(1.0 / width, 1.0 / height);
    
    // 1. 取得中心點的深度與色彩引導值
    float depthCenter = texture(depthTex, TexCoords).r;
    float guideCenter = getGuideValue(TexCoords);

    // 2. 計算中心點的色彩梯度強度
    float gradCenter = getGuideGradient(TexCoords);

    // 3. 梯度自適應調整 sigmaColor
    // 如果梯度低於門檻（代表這不是強強烈的幾何邊緣，可能是平坦區或細碎紋理），放寬 3 倍色彩容忍度以強制平滑
    float adaptiveSigmaColor = sigmaColor;
    if (gradCenter < edgeThreshold) {
        adaptiveSigmaColor *= 3.0f;
    }

    float weightSum = 0.0;
    float depthSum = 0.0;
    int kernelRadius = 6; // 9x9 採樣視窗

    // 4. 雙迴圈進行雙邊加權
    for(int y = -kernelRadius; y <= kernelRadius; y++) {
        for(int x = -kernelRadius; x <= kernelRadius; x++) {
            vec2 offset = vec2(float(x), float(y)) * tex_offset;
            vec2 sampleUV = TexCoords + offset;
            
            if(sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) continue;

            float sampleDepth = texture(depthTex, sampleUV).r;
            float sampleGuide = getGuideValue(sampleUV);
            
            // 空間權重
            float dist2 = float(x*x + y*y);
            float spatialWeight = exp(-dist2 / (2.0 * sigmaSpatial * sigmaSpatial));
            
            // 範圍權重（使用梯度修正後的 adaptiveSigmaColor）
            float colorDist = sampleGuide - guideCenter;
            float colorWeight = exp(-(colorDist * colorDist) / (2.0 * adaptiveSigmaColor * adaptiveSigmaColor));
            
            float w = spatialWeight * colorWeight;
            
            depthSum += sampleDepth * w;
            weightSum += w;
        }
    }
    
    FilteredDepth = depthSum / weightSum;
}