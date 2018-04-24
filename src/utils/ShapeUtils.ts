import {Conv2dOptions} from "../op/cnn/Conv2dOp";
import TensorFlags from "../tensor/TensorFlags";

export default class ShapeUtils {

  /**
   * Broadcast two shapes, return broadcasted shape
   */
  static broadcastShapes(a: number[], b: number[]): number[] {
    let rank = Math.max(a.length, b.length);
    let result = new Array(rank); //uninitialized

    let aIndex = a.length - 1;
    let bIndex = b.length - 1;

    for (let dim = rank - 1; dim >= 0; dim--) {
      let left = aIndex >= 0 ? a[aIndex] : 1;
      let right = bIndex >= 0 ? b[bIndex] : 1;

      if (left !== 1 && right !== 1 && left !== right) {
        throw new Error("cannot broadcast shapes." + a + ", " + b + " not compatible");
      }

      result[dim] = Math.max(left, right);
      aIndex--;
      bIndex--;
    }

    return result;
  }

  /**
   * Check to see if shape can be broadcasted to newShape
   * @param {number[]} shape
   * @param {number[]} newShape
   */
  static canBroadcastTo(shape: number[], newShape: number[]): boolean {
    if (shape.length > newShape.length) {
      return false;
    }

    let i = newShape.length - 1;
    let j = shape.length - 1;
    for (; i >= 0 && j >= 0; i--, j--) {
      let a = newShape[i];
      let b = shape[j];

      if (a !== 1 && b !== 1 && a !== b) {
        return false;
      }
    }

    return true;
  }

  static computeConv2dShape(imageShape: number[], kernelShape: number[], options: Conv2dOptions) {
    let outputHeight = ShapeUtils.computeConvOutSize(imageShape[2], kernelShape[2], options.padHeight, options.strideHeight);
    let outputWidth = ShapeUtils.computeConvOutSize(imageShape[3], kernelShape[3], options.padWidth, options.strideWidth);
    return [imageShape[0], kernelShape[0], outputHeight, outputWidth];
  }

  static computeConvOutSize(imageSize: number, kernelSize: number, padSize = 0, stride = 1) {
    let result = (imageSize - kernelSize + 2 * padSize) / stride + 1;
    if (result !== Math.floor(result)) {
      throw new Error("Cannot do conv with these values: imageSize: {" + imageSize + "}, kernelSize: {" + kernelSize + "}");
    }
    return result;
  }

  /**
   * Returns the indices that are broadcasted
   * @param {number[]} shape
   * @param {number[]} newShape
   * @returns {number[]}
   */
  static getBroadcastIndices(shape: number[], newShape: number[]): number[] {
    let indices: number[] = [];
    let i = newShape.length - 1;
    let j = shape.length - 1;
    for (; i >= 0 && j >= 0; i--, j--) {
      let a = newShape[i];
      let b = shape[j];

      if (b === 1 && a !== 1) {
        indices.push(i);
      }
    }

    for (; i >= 0; i--) {
      indices.push(i);
    }

    return indices;
  }

  /**
   * Get the shape that must be reshaped to match result's rank
   */
  static getBroadcastedShape(input: number[], result: number[]): number[] {
    if (input.length >= result.length) {
      return input;
    }

    let newShape = input.slice();
    for (let i = 0; i < result.length - input.length; i++) {
      newShape.unshift(1);
    }
    return newShape;
  }

  static getLength(shape: number[]): number {
    let mul = 1;
    for (let dim of shape) {
      mul *= dim;
    }
    return mul;
  }

  static getReducedDims(shape: number[], dims: number | number[]): boolean[] {
    let reducedDims: boolean[] = [];
    for (let i = 0; i < shape.length; i++) {
      reducedDims.push(false);
    }

    if (Array.isArray(dims)) {
      for (let dim of dims) {
        reducedDims[dim] = true;
      }
    } else {
      if (dims === -1) {
        for (let i = 0; i < shape.length; i++) {
          reducedDims[i] = true;
        }
      } else {
        reducedDims[dims] = true;
      }
    }
    return reducedDims;
  }

  /**
   * Get the indices that are reduced, return undefined if one of the indices is not reduced.
   */
  static getReductionIndices(a: number[], b: number[]): { left: number[] | undefined, right: number[] | undefined } {
    let resultShape = ShapeUtils.broadcastShapes(a, b);

    let aBroad = ShapeUtils.getBroadcastedShape(a, resultShape);
    let bBroad = ShapeUtils.getBroadcastedShape(b, resultShape);

    let left = [];
    let right = [];
    for (let i = 0; i < resultShape.length; i++) {
      if (aBroad[i] === 1 && aBroad[i] !== resultShape[i]) {
        left.push(i);
      }

      if (bBroad[i] === 1 && bBroad[i] !== resultShape[i]) {
        right.push(i);
      }
    }

    return {
      left: left.length > 0 ? left : undefined,
      right: right.length > 0 ? right : undefined
    };
  }

  static getSlices(shape: number[], dimension: number): number {
    let slices = 1;
    for (let i = 0; i < shape.length; i++) {
      slices *= (i === dimension) ? 1 : shape[i];
    }
    return slices;
  }

  static getStrides(shape: number[]): number[] {
    let rank = shape.length;
    let strides = new Array(rank);

    let val = 1;
    for (let i = rank - 1; i >= 0; --i) {
      strides[i] = val;
      val *= shape[i];
    }

    return strides;
  }

  static inferOrder(shape: number[], strides: number[]): string {
    let isFortran = true; // Fortran Contiguous
    let isC = true; // C Contiguous

    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      if (stride !== strides[i]) {
        isC = false;
        break;
      }
      stride *= shape[i];
    }

    stride = 1;
    for (let i = 0; i < shape.length; i++) {
      if (stride !== strides[i]) {
        isFortran = false;
        break;
      }
      stride *= shape[i];
    }

    if (isFortran && !isC) {
      return "f";
    }
    return "c";
  }

  /**
   * This method handles shape dimension == -1
   */
  static inferShape(length: number, shape: number[]) {
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] == -1) {
        let prod = 1;
        for (let j = 0; j < shape.length; j++) {
          if (j !== i && shape[j] > 1) {
            prod *= shape[j];
          }
        }

        shape[i] = length / prod;
      }
    }
    return shape;
  }

  static invertPermutation(axis: number[]): number[] {
    let newAxis = axis.slice();
    for (let i = 0; i < newAxis.length; i++) {
      newAxis[axis[i]] = i;
    }
    return newAxis;
  }

  static reduce(shape: number[], dimension: number): number[] {
    let result = shape.slice();
    for (let i = 0; i < shape.length; i++) {
      if (i === dimension || dimension === -1) {
        result[i] = 1;
      }
    }
    return result;
  }

  /**
   * Get the reduced shape
   *
   * @param {[int]} shape
   * @param {int | [int]} dims - dims to reduce
   * @param {boolean} [keepDims = false]
   * @returns {[int]} Reduced Shape
   */
  static reduceShape(shape: number[], dims: number | number[], keepDims: boolean): number[] {
    let resultShape = [];
    let reducedDims = ShapeUtils.getReducedDims(shape, dims);

    for (let i = 0; i < shape.length; i++) {
      if (!reducedDims[i]) {
        resultShape.push(shape[i]);
      } else if (keepDims) {
        resultShape.push(1);
      }
    }

    return resultShape;
  }

  /**
   * By calling this method the caller ensures the two shapes have the same rank
   * and can be cleanly divided.
   * Normally used on reduction.
   */
  static safeDivide(shape1: number[], shape2: number[]): number[] {
    let result = [];
    for (let i = 0; i < shape1.length; i++) {
      result.push(shape1[i] / shape2[i]);
    }
    return result;
  }

  static shapeEquals(a: number[], b: number[]): boolean {
    if (a == null || b == null || a.length != b.length) {
      return false;
    }
    for (let i = 0; i < a.length; i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }

  static transposeShape(shape: number[], newAxis: number[] = []): number[] {
    let rank = shape.length;
    let newShape = new Array(rank);
    if (newAxis.length === 0) {
      newAxis = new Array(rank);
      for (let i = 0; i < newAxis.length; i++) {
        newAxis[i] = rank - 1 - i;
      }
    }

    for (let i = 0; i < rank; i++) {
      let axis = newAxis[i];
      newShape[i] = shape[axis];
    }

    return newShape;
  }

  static inferFlags(shape: number[], strides: number[] | undefined, offset: number): TensorFlags {
    return {
      cContiguous: true,
      fContiguous: true
    }
  }
}