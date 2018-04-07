import Shape from "./Shape";
import TensorMath from "./TensorMath";
import TensorFactory from "./utils/TensorFactory";
import TensorFormat from "./utils/TensorFormat";
import TensorUtils from "./utils/TensorUtils";

export default class Tensor {

  static tensorFormat: TensorFormat = new TensorFormat();

  private readonly _data: Float32Array;
  private readonly _offset: number;
  private readonly _shape: Shape;

  get data() {
    return this._data;
  }

  get isMatrix() {
    return this.rank === 2;
  }

  get isScalar() {
    return this.rank === 0;
  }

  get isVector() {
    return this.rank === 1;
  }

  get length() {
    return this._shape.length;
  }

  get offset() {
    return this._offset;
  }

  get rank() {
    return this._shape.rank;
  }

  get shape() {
    return this._shape.shape;
  }

  get slices() {
    return this.shape[0];
  }

  get strides() {
    return this._shape.strides;
  }

  constructor(data: Float32Array, shape: Shape, offset = 0) {
    this._data = data;
    this._shape = shape;
    this._offset = offset;
  }

  /**
   * arange function. Note that first parameter is stop, not start, due to how JS params works.
   */
  static arange(stop: number, start: number = 0, step: number = 1) {
    return TensorFactory.arange(stop, start, step);
  }

  static create(array: number): Tensor;

  static create(array: number[]): Tensor;

  static create(array: number[][]): Tensor;

  static create(array: number[][][]): Tensor;

  static create(array: number[][][][]): Tensor;

  static create(array: any): Tensor {
    return TensorFactory.create(array);
  }

  static linspace(start: number, stop: number, num: number) {
    return TensorFactory.linspace(start, stop, num);
  }

  static ones(shape: number[]) {
    return TensorFactory.ones(shape);
  }

  static rand(shape: number[]): Tensor {
    return TensorFactory.rand(shape);
  }

  static scalar(scalar: number): Tensor {
    return TensorFactory.scalar(scalar);
  }

  static zeros(shape: number[]): Tensor {
    return TensorFactory.zeros(shape);
  }

  abs(): Tensor {
    return TensorMath.abs(this);
  }

  absi(): Tensor {
    return TensorMath.abs(this, this);
  }

  add(other: Tensor): Tensor {
    return TensorMath.add(this, other);
  }

  addi(other: Tensor): Tensor {
    return TensorMath.add(this, other, this);
  }

  broadcast(shape: number[]): Tensor {
    return TensorUtils.broadcastTensor(this, shape);
  }

  ceil(): Tensor {
    return TensorMath.ceil(this);
  }

  ceili(): Tensor {
    return TensorMath.ceil(this, this);
  }

  divide(other: Tensor): Tensor {
    return TensorMath.divide(this, other);
  }

  dividei(other: Tensor): Tensor {
    return TensorMath.divide(this, other, this);
  }

  equal(other: Tensor): Tensor {
    return TensorMath.equal(this, other);
  }

  erf(): Tensor {
    return TensorMath.erf(this);
  }

  erfc(): Tensor {
    return TensorMath.erfc(this);
  }

  fill(scalar: number): Tensor {
    return TensorMath.fill(this, scalar);
  }

  filli(scalar: number): Tensor {
    return TensorMath.fill(this, scalar, this);
  }

  floor(): Tensor {
    return TensorMath.floor(this);
  }

  floorDiv(other: Tensor): Tensor {
    return TensorMath.floorDiv(this, other);
  }

  floorMod(other: Tensor): Tensor {
    return TensorMath.floorMod(this, other);
  }

  floori(): Tensor {
    return TensorMath.floor(this, this);
  }

  get(indices: number | number[]): number {
    if (!Array.isArray(indices)) {
      indices = this._shape.getIndices(indices);
    }
    let offset = this._shape.getOffset(indices) + this.offset;
    return this._data[offset];
  }

  greater(other: Tensor): Tensor {
    return TensorMath.greater(this, other);
  }

  greaterEqual(other: Tensor): Tensor {
    return TensorMath.greaterEqual(this, other);
  }

  less(other: Tensor): Tensor {
    return TensorMath.less(this, other);
  }

  lessEqual(other: Tensor): Tensor {
    return TensorMath.lessEqual(this, other);
  }

  matmul(other: Tensor): Tensor {
    return TensorMath.matmul(this, other);
  }

  multiply(other: Tensor): Tensor {
    return TensorMath.multiply(this, other);
  }

  negate() {
    return TensorMath.negate(this);
  }

  negatei() {
    return TensorMath.negate(this, this);
  }

  notEqual(other: Tensor): Tensor {
    return TensorMath.notEqual(this, other);
  }

  reciprocal(): Tensor {
    return TensorMath.reciprocal(this);
  }

  reciprocalGrad(): Tensor {
    return TensorMath.reciprocalGrad(this);
  }

  reciprocalGradi(): Tensor {
    return TensorMath.reciprocalGrad(this, this);
  }

  reciprocali(): Tensor {
    return TensorMath.reciprocal(this, this);
  }

  repeat(multiple: number, dimension: number = -1): Tensor {
    return TensorMath.repeat(this, multiple, dimension);
  }

  reshape(shape: number[]): Tensor {
    return TensorUtils.reshape(this, shape);
  }

  round(): Tensor {
    return TensorMath.round(this);
  }

  roundi(): Tensor {
    return TensorMath.round(this, this);
  }

  set(indices: number | number[], value: number): void {
    if (!Array.isArray(indices)) {
      indices = this._shape.getIndices(indices);
    }
    let offset = this._shape.getOffset(indices) + this.offset;
    this._data[offset] = value;
  }

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
  slice(begin: number[], size?: number[]): Tensor {
    let offset = this.offset;
    let newShape = this.shape.slice();
    if (!size) {
      size = new Array(this.rank).fill(-1);
    }

    for (let i = 0; i < this.rank; i++) {
      let a = begin[i] < 0 ? begin[i] + this.shape[i] : begin[i];
      offset += a * this.strides[i];

      let width = size[i] < 0 ? (this.shape[i] - a) : Math.min(this.shape[i] - a, size[i]);
      newShape[i] = width;
    }

    let shape = new Shape(newShape, this.strides, this._shape.order);
    return new Tensor(this._data, shape, offset);
  }

  slice2(num: number): Tensor {
    let offset = this.offset;
    let newShape = [];
    let newStrides = [];

    offset += num * this.strides[0];

    for (let i = 1; i < this.rank; i++) {
      newShape.push(this.shape[i]);
      newStrides.push(this.strides[i]);
    }

    let shape = new Shape(newShape, newStrides, this._shape.order);
    return new Tensor(this._data, shape, offset);
  }

  subtract(other: Tensor): Tensor {
    return TensorMath.subtract(this, other);
  }

  tile(repeats: number[]): Tensor {
    return TensorMath.tile(this, repeats);
  }

  toString() {
    return Tensor.tensorFormat.format(this);
  }

  truncDiv(other: Tensor): Tensor {
    return TensorMath.truncDiv(this, other);
  }

  truncMod(other: Tensor): Tensor {
    return TensorMath.truncMod(this, other);
  }

}