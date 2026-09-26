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

void main()
{
    vec2 tex_offset = vec2(1.0 / width, 1.0 / height);
    
    // 1. 取得中心點的深度
    float depthCenter = texture(depthTex, TexCoords).r;
    
    // 2. 取得中心點的色彩引導 (Guide)
    // 為了效能與邊緣保留，我們只取亮度或紅色通道作為引導
    float guideCenter = 0.0;
    if(isYCbCr > 0.5f){
        // 如果是 YUV (例如 MP4 影片)，讀取 Y 通道
        vec2 texcoord_Y = vec2(TexCoords.x, TexCoords.y * height / (height * 1.5f + chroma_offset));
        guideCenter = texture(colorTex, texcoord_Y).r;
    } else {
        // 如果是 RGB (例如 PNG)，讀取 R 通道
        guideCenter = texture(colorTex, TexCoords).r;
    }

    float weightSum = 0.0;
    float depthSum = 0.0;

    // 定義採樣視窗半徑，例如 4 代表 9x9 的視窗
    int kernelRadius = 4; 

    // 雙迴圈進行鄰域採樣
    for(int y = -kernelRadius; y <= kernelRadius; y++) {
        for(int x = -kernelRadius; x <= kernelRadius; x++) {
            vec2 offset = vec2(float(x), float(y)) * tex_offset;
            vec2 sampleUV = TexCoords + offset;
            
            // 邊界檢查，避免採樣到畫面外
            if(sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) continue;

            // 讀取鄰域的深度
            float sampleDepth = texture(depthTex, sampleUV).r;
            
            // 讀取鄰域的色彩引導
            float sampleGuide = 0.0;
            if(isYCbCr > 0.5f){
                vec2 sample_Y = vec2(sampleUV.x, sampleUV.y * height / (height * 1.5f + chroma_offset));
                sampleGuide = texture(colorTex, sample_Y).r;
            } else {
                sampleGuide = texture(colorTex, sampleUV).r;
            }
            
            // 計算空間權重 (Spatial Weight)
            float dist2 = float(x*x + y*y);
            float spatialWeight = exp(-dist2 / (2.0 * sigmaSpatial * sigmaSpatial));
            
            // 計算範圍/色彩權重 (Range Weight)
            float colorDist = sampleGuide - guideCenter;
            float colorWeight = exp(-(colorDist * colorDist) / (2.0 * sigmaColor * sigmaColor));
            
            // 聯合權重
            float w = spatialWeight * colorWeight;
            
            depthSum += sampleDepth * w;
            weightSum += w;
        }
    }
    
    // 輸出濾波後的深度值
    FilteredDepth = depthSum / weightSum;
}