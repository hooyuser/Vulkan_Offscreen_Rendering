IF NOT EXIST "shaders/NUL" mkdir "shaders"
D:/VulkanSDK/1.2.170.0/Bin32/glslc.exe glsl_shaders/shader.vert -o shaders/vert.spv
D:/VulkanSDK/1.2.170.0/Bin32/glslc.exe glsl_shaders/shader.frag -o shaders/frag.spv
pause