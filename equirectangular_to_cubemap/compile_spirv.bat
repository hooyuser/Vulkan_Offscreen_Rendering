IF NOT EXIST "shaders/NUL" mkdir "shaders"
C:/VulkanSDK/1.2.182.0/Bin32/glslc.exe glsl_shaders/shader.vert -o shaders/vert.spv
C:/VulkanSDK/1.2.182.0/Bin32/glslc.exe glsl_shaders/shader.frag -o shaders/frag.spv
pause