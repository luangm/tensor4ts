import {ArrayLike} from "../Tensor";
import ShapeUtils from "./ShapeUtils";

export default class ArrayUtils {

  static findShape(array: any): number[] {
    if (!Array.isArray(array)) {
      return [];
    }

    if (array.length === 0) {
      throw new Error("array's length cannot be zero");
    }

    let shape = ArrayUtils.findShape(array[0]);
    shape.unshift(array.length);
    return shape;
  }

  static flatten(array: any, buffer: ArrayLike, shape: number[]) {
    let indices: number[] = new Array(shape.length).fill(0);
    let strides = ShapeUtils.getStrides(shape);
    this.flattenInternal(array, buffer, strides, indices, 0);
  }

  private static flattenInternal(array: any, buffer: ArrayLike, strides: number[], indices: number[], dim: number = 0) {
    for (let i = 0; i < array.length; i++) {
      let value = array[i];
      indices[dim] = i;
      if (Array.isArray(value)) {
        ArrayUtils.flattenInternal(value, buffer, strides, indices, dim + 1);
      } else {
        let offset = this.getOffset(strides, indices);
        buffer[offset] = value;
      }
    }
  }

  private static getOffset(strides: number[], indices: number[]): number {
    let offset = 0;
    for (let i = 0; i < strides.length; i++) {
      offset += strides[i] * indices[i];
    }
    return offset;
  }

}