import {DataType} from "./DataType";

export interface ArrayLike {
  [index: number]: number;
}

/**
 * This defines the interface to Tensor instances.
 * Individual implementations should override methods to handle different data types.
 */
export default interface Tensor {

  readonly data: ArrayLike;

  /**
   * The type of Data stored in the Tensor
   */
  readonly dataType: DataType;

  /**
   * The number of items in this Tensor
   */
  readonly length: number;

  /**
   * Offset in the data. Typically comes from an slice op.
   */
  readonly offset: number;

  /**
   * c or f
   */
  readonly order: string;

  /**
   * The rank of the Tensor
   */
  readonly rank: number;

  /**
   * The Shape of the Tensor
   */
  readonly shape: number[];

  /**
   *
   */
  readonly strides: number[];

  get(indices: number[]): number;

  reshape(shape: number[]): Tensor;

  set(indices: number[], value: number): void;

  /**
   * Slice the tensor.
   *
   * if any begin is -ve, it means from the back.
   * If size is not specified or any size's dimension is -1. it means to the end.
   *
   * @param {number[]} begin - the indices to start the slice
   * @param {number[]} size - the size(shape) of the slice.
   * @returns {Tensor}
   */
  slice(begin: number[], size?: number[]): Tensor;

  sliceSingle(num: number): Tensor;

  // abs(): Tensor;
  //
  // add(other: Tensor): Tensor;
  //
  // broadcast(shape: number[]): Tensor;
  //
  // ceil(): Tensor;
  //
  // divide(other: Tensor): Tensor;
  //
  // dup(): Tensor;
  //
  // equal(other: Tensor): Tensor;
  //
  // erf(): Tensor;
  //
  // erfc(): Tensor;
  //
  // exp(): Tensor;
  //
  // fill(scalar: number): Tensor;
  //
  // floor(): Tensor;
  //
  // floorDiv(other: Tensor): Tensor;
  //
  // floorMod(other: Tensor): Tensor;
  //

  //
  // greater(other: Tensor): Tensor;
  //
  // greaterEqual(other: Tensor): Tensor;
  //
  // less(other: Tensor): Tensor;
  //
  // lessEqual(other: Tensor): Tensor;
  //
  // log(): Tensor;
  //
  // matmul(other: Tensor, transposeLeft: boolean, transposeRight: boolean): Tensor;
  //
  // multiply(other: Tensor): Tensor;
  //
  // negate(): Tensor;
  //
  // notEqual(other: Tensor): Tensor;
  //
  // reciprocal(): Tensor;
  //
  // reciprocalGrad(): Tensor;
  //
  // reduceLogSumExp(dims: number | number[], keepDims: boolean): Tensor;
  //
  // reduceMax(dims: number | number[], keepDims: boolean): Tensor;
  //
  // reduceMean(dims: number | number[], keepDims: boolean): Tensor;
  //
  // reduceMin(dims: number | number[], keepDims: boolean): Tensor;
  //
  // reduceProd(dims: number | number[], keepDims: boolean): Tensor;
  //
  // reduceSum(dims: number | number[], keepDims: boolean): Tensor;
  //
  // repeat(multiple: number, dimension: number): Tensor;
  //

  //
  // round(): Tensor;
  //

  //
  // subtract(other: Tensor): Tensor;
  //
  // tile(repeats: number[]): Tensor;
  //
  // transpose(newAxis: number[]): Tensor;
  //
  // truncDiv(other: Tensor): Tensor;
  //
  // truncMod(other: Tensor): Tensor;

}