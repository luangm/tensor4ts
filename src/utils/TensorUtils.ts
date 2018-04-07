import Tensor from "../Tensor";
import Shape from "../Shape";
import ShapeUtils from "./ShapeUtils";

export default class TensorUtils {

  static broadcastTensor(tensor: Tensor, shape: number[]): Tensor {
    if (TensorUtils.shapeEquals(tensor.shape, shape)) {
      return tensor;
    }

    // if we're on scalar, we can just create new array
    // if (this.isScalar())
    //   return Nd4j.createUninitialized(shape).assign(this.getDouble(0));

    let compatible = this.shapeIsCompatible(tensor.shape, shape);

    if (!compatible) {
      throw new Error('Incompatible broadcast from ' + tensor.shape + ' to ' + shape);
    }

    let retShape = ShapeUtils.broadcastShapes(tensor.shape, shape);

    // Shortcut for Zeros
    if (tensor.isZeros) {
      return Tensor.sparseZeros(retShape);
    }

    let result = Tensor.zeros(retShape);
    let broadcastDims: number[] = [];

    // pad front
    let front = retShape.length - tensor.shape.length;
    for (let i = 0; i < retShape.length; i++) {
      if (i < front || (tensor.shape[i - front] === 1 && retShape[i] !== 1)) {
        broadcastDims.push(1);
      } else {
        broadcastDims.push(0);
      }
    }

    let indices = new Array(retShape.length).fill(0);
    broadcastAtDim(0, indices);

    function broadcastAtDim(dim: number, targetIndices: number[]) {
      if (dim === targetIndices.length) {
        let sourceIndices = targetIndices.slice(front);
        for (let i = targetIndices.length - 1; i >= front; i--) {
          if (broadcastDims[i] === 1) {
            sourceIndices[i - front] = 0;
          }
        }
        let sourceOffset = TensorUtils.computeOffset(sourceIndices, tensor.shape, tensor.strides);
        let targetOffset = TensorUtils.computeOffset(targetIndices, result.shape, result.strides);
        // console.log(sourceOffset, tensor.data[sourceOffset], targetOffset);
        result.data[targetOffset] = tensor.data[sourceOffset];
        return;
      }

      for (let i = 0; i < retShape[dim]; i++) {
        targetIndices[dim] = i;
        broadcastAtDim(dim + 1, targetIndices);
      }
    }

    return result;
  }


  //
  // static computeConv2dShape(image, kernel) {
  //     let numImages = image.shape[0];
  //     let channels = image.shape[1];
  //     let height = image.shape[2]; // rows
  //     let width = image.shape[3]; // cols
  //
  //     let numKernels = kernel.shape[0];
  //     let kernelChannels = kernel.shape[1];
  //     let kernelHeight = kernel.shape[2]; // rows
  //     let kernelWidth = kernel.shape[3]; // cols
  //
  //     let outputHeight = TensorUtils.computeConv2dOutSize(height, kernelHeight);
  //     let outputWidth = TensorUtils.computeConv2dOutSize(width, kernelWidth);
  //
  //     return [numImages, numKernels, outputHeight, outputWidth];
  // }
  //
  // static computeMaxPoolShape(imageShape, kernelShape, strideWidth, strideHeight) {
  //     let numImages = imageShape[0];
  //     let channels = imageShape[1];
  //     let height = imageShape[2]; // rows
  //     let width = imageShape[3]; // cols
  //
  //     let numKernels = kernelShape[0];
  //     let kernelChannels = kernelShape[1];
  //     let kernelHeight = kernelShape[2]; // rows
  //     let kernelWidth = kernelShape[3]; // cols
  //
  //     let outputHeight = TensorUtils.computeConv2dOutSize(height, kernelHeight, 0, strideHeight);
  //     let outputWidth = TensorUtils.computeConv2dOutSize(width, kernelWidth, 0, strideWidth);
  //
  //     return [numImages, numKernels, outputHeight, outputWidth];
  // }

  static computeOffset(indices: number[], shape: number[], strides: number[]): number {
    let offset = 0;
    for (let i = 0; i < shape.length; i++) {
      offset += indices[i] * strides[i];
    }
    return offset;
  }

  static reshape(tensor: Tensor, newShape: number[]): Tensor {
    for (let i = 0; i < newShape.length; i++) {
      if (newShape[i] == -1) {

        let prod = 1;
        for (let j = 0; j < newShape.length; j++) {
          if (j !== i) {
            prod *= newShape[j];
          }
        }

        newShape[i] = tensor.length / prod;
      }
    }

    if (TensorUtils.shapeEquals(tensor.shape, newShape)) {
      return tensor;
    }

    return new Tensor(tensor.data, new Shape(newShape), tensor.offset, tensor.isZeros);
  }

  static shapeEquals(a: number[], b: number[]): boolean {
    if (a == null || b == null) {
      return false;
    }
    if (a === b) {
      return true;
    }
    if (a.length !== b.length) {
      return false;
    }

    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) {
        return false;
      }
    }

    return true;
  }

  static shapeIsCompatible(a: number[], b: number[]): boolean {
    let rank = Math.max(a.length, b.length);
    let aIndex = a.length - 1;
    let bIndex = b.length - 1;

    for (let i = 0; i < rank; i++) {
      if (aIndex < 0 || bIndex < 0) {
        break;
      }

      let left = a[aIndex--];
      let right = b[bIndex--];

      if (left !== 1 && right !== 1 && left !== right) {
        return false;
      }
    }

    return true;
  }
}
