import Tensor from "./Tensor";
import FloatTensor from "./tensor/FloatTensor";
import TensorMath from "./TensorMath";
import ArrayUtils from "./utils/ArrayUtils";
import ShapeUtils from "./utils/ShapeUtils";

export default class TensorFactory {

  static arange(stop: number, start: number = 0, step: number = 1): Tensor {
    return TensorMath.arange(stop, start, step);
  }

  /**
   * Create a tensor from the given array.
   * The length of the tensor is array.length * array[0].length * .... array[0]...[0].length
   * If any array's dimension does not match, an error is thrown.
   */
  static create(array: any): Tensor {
    let shape = ArrayUtils.findShape(array);
    let length = ShapeUtils.getLength(shape);
    let buffer = new Float32Array(length);
    ArrayUtils.flatten(array, buffer, shape);
    return new FloatTensor(buffer, shape);
  }

  // static linspace(start: number, stop: number, num: number) {
  //   return TensorMath.linspace(start, stop, num);
  // }
  //
  // static ones(shape: number[]): Tensor {
  //   return TensorFactory.zeros(shape);
  // }
  //
  // static rand(shape: number[]): Tensor {
  //   let tensor = TensorFactory.zeros(shape);
  //   return TensorMath.rand(tensor, tensor);
  // }

  static scalar(scalar: number): Tensor {
    let data = new Float32Array([scalar]);
    return new FloatTensor(data, []);
  }

  // static sparseZeros(shape: number[]): Tensor {
  //   let shapeObj = new Shape(shape);
  //   let data = new Float32Array(0);
  //   return new FloatTensor(data, shapeObj, 0, true);
  // }

  static vector(array: number[]): Tensor {
    let data = new Float32Array(array);
    return new FloatTensor(data, [array.length]);
  }

  static zeros(shape: number[]): Tensor {
    let length = ShapeUtils.getLength(shape);
    let data = new Float32Array(length);
    return new FloatTensor(data, shape);
  }

}