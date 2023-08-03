
const main = async(size) => {
    if (!("gpu" in navigator)) {
        console.log(
            "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
        );
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log("Failed to get GPU adapter.");
      return;
    }
    const device = await adapter.requestDevice();

    const shaderModule = device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> firstArray : array<u32>;
          @group(0) @binding(1) var<storage, read> secondArray : array<u32>;
          @group(0) @binding(2) var<storage, read_write> resultArray : array<u32>;
      
          @compute @workgroup_size(1)
          fn main(@builtin(global_invocation_id) global_id : vec3u) {
            // Guard against out-of-bounds work group sizes
            if (global_id.x >= arrayLength(&firstArray)) {
              return;
            }
    
            resultArray[global_id.x] = firstArray[global_id.x] * secondArray[global_id.x];
          }
        `
      });
      
    const gpuBufferVector1 = device.createBuffer({
        mappedAtCreation: true,
        size: size,
        usage: GPUBufferUsage.STORAGE 
    });
    const arraybufferArray1 = gpuBufferVector1.getMappedRange();
    let arr = new Uint32Array(arraybufferArray1);
    for (let i = 0; i < size; i++) {
        arr[i] = i;
    }
    gpuBufferVector1.unmap();

    const gpuBufferVector2 = device.createBuffer({
        mappedAtCreation: true,
        size: size,
        usage: GPUBufferUsage.STORAGE
    });
    const arraybufferArray2 = gpuBufferVector2.getMappedRange();
    let arr2 = new Uint32Array(arraybufferArray2);
    for (let i = 0; i < size; i++) {
        arr2[i] = i;
    }
    gpuBufferVector2.unmap();

    const gpuBufferVector3 = device.createBuffer({
        mappedAtCreation: false,
        size: size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: gpuBufferVector1
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: gpuBufferVector2
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: gpuBufferVector3
                }
            }
        ]
    });

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
          module: shaderModule,
          entryPoint: "main"
        }
      });
    
    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupCountX = Math.ceil(size);
    passEncoder.dispatchWorkgroups(workgroupCountX);
    passEncoder.end();

    const gpuReadBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
    
    commandEncoder.copyBufferToBuffer(
        gpuBufferVector3 /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        size /* size */
    );

    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange();
    return new Uint32Array(arrayBuffer);
}