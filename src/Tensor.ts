import {DataType} from "./DataType";

export interface TensorBufferLike {
  readonly BYTES_PER_ELEMENT: number;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly length: number;

  copyWithin(target: number, start: number, end?: number): this;

  fill(value: number, start?: number, end?: number): this;

  [index: number]: number;
}

/**
 * This defines the interface to Tensor instances.
 * Individual implementations should override methods to handle different data types.
 */
export default interface Tensor {

  /**
   * Whether the tensor is C-contiguous
   */
  readonly cContiguous: boolean;

  /**
   * This is used to access the data array directly.
   * This is a common interface for TypedArrays.
   */
  readonly data: TensorBufferLike;

  /**
   * The type of Data stored in the Tensor
   */
  readonly dataType: DataType;
  /**
   * Whether the tensor is F-contiguous
   */
  readonly fContiguous: boolean;
  /**
   * The number of items in this Tensor
   */
  readonly length: number;
  /**
   * Offset in the data. Typically comes from an slice op.
   */
  readonly offset: number;
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

  /**
   * Add other with self. Return a new Tensor
   */
  add(other: Tensor): Tensor;

  /**
   * Broadcast the tensor to the new Shape
   */
  broadcast(newShape: number[]): Tensor;

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