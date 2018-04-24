import {DataType} from "./DataType";
import Tensor from "./Tensor";
import FloatTensor from "./tensor/FloatTensor";
import IntTensor from "./tensor/IntTensor";
import ArrayUtils from "./utils/ArrayUtils";
import ShapeUtils from "./utils/ShapeUtils";

const TYPE_MAP: Map<string, DataType> = new Map([
  [DataType.Int32, DataType.Int32],
  [DataType.Float32, DataType.Float32],
  [DataType.Int32 + DataType.Int32, DataType.Int32],
  [DataType.Float32 + DataType.Int32, DataType.Float32],
  [DataType.Int32 + DataType.Float32, DataType.Float32],
  [DataType.Float32 + DataType.Float32, DataType.Float32]
]);

export default class TensorFactory {

  //
  // static arange(stop: number, start: number = 0, step: number = 1): Tensor {
  //   return TensorMath.arange(stop, start, step);
  // }

  /**
   * Create a tensor from the given array.
   * The length of the tensor is array.length * array[0].length * .... array[0]...[0].length
   * If any array's dimension does not match, an error is thrown.
   */
  static create(array: any, dataType: DataType = DataType.Float32): Tensor {
    let shape = ArrayUtils.findShape(array);
    let length = ShapeUtils.getLength(shape);

    if (dataType === DataType.Int32) {
      let buffer = new Int32Array(length);
      ArrayUtils.flatten(array, buffer, shape);
      return new IntTensor(buffer, shape);
    } else {
      let buffer = new Float32Array(length);
      ArrayUtils.flatten(array, buffer, shape);
      return new FloatTensor(buffer, shape);
    }
  }

  static empty(shape: number[], aType: DataType, bType?: DataType): Tensor {
    let length = ShapeUtils.getLength(shape);
    let arr: string = bType ? aType + bType : aType;
    let combined = TYPE_MAP.get(arr);
    switch (combined) {
      case DataType.Int32:
        return new IntTensor(new Int32Array(length), shape);
      default:
        return new FloatTensor(new Float32Array(length), shape);
    }
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

  // static sparseZeros(shape: number[]): Tensor {
  //   let shapeObj = new Shape(shape);
  //   let data = new Float32Array(0);
  //   return new FloatTensor(data, shapeObj, 0, true);
  // }

}