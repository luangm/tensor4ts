import Shape from "./Shape";
import TensorMath from "./TensorMath";
import TensorFactory from "./utils/TensorFactory";
import TensorFormat from "./utils/TensorFormat";
import TensorUtils from "./utils/TensorUtils";

export default class Tensor {

  static tensorFormat: TensorFormat = new TensorFormat();
  private _data: Float32Array;
  private _offset: number;
  private _shape: Shape;

  constructor(data: Float32Array, shape: Shape, offset = 0) {
    this._data = data;
    this._shape = shape;
    this._offset = offset;
  }

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

  static linspace(start: number, stop: number, num: number = 1) {
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

  add(other: Tensor): Tensor {
    return TensorMath.add(this, other);
  }

  addi(other: Tensor): Tensor {
    return TensorMath.add(this, other, this);
  }

  broadcast(shape: number[]): Tensor {
    return TensorUtils.broadcastTensor(this, shape);
  }

  divide(other: Tensor): Tensor {
    return TensorMath.divide(this, other);
  }

  dividei(other: Tensor): Tensor {
    return TensorMath.divide(this, other, this);
  }

  fill(scalar: number): Tensor {
    return TensorMath.fill(this, scalar);
  }

  filli(scalar: number): Tensor {
    return TensorMath.fill(this, scalar, this);
  }

  get(indices: number[]): number {
    let offset = this._shape.getOffset(indices) + this.offset;
    return this._data[offset];
  }

  matmul(other: Tensor): Tensor {
    return TensorMath.matmul(this, other);
  }

  mod(other: Tensor): Tensor {
    return TensorMath.mod(this, other);
  }

  multiply(other: Tensor): Tensor {
    return TensorMath.multiply(this, other);
  }

  negate() {
    return TensorMath.negate(this);
  }

  reshape(shape: number[]): Tensor {
    return TensorUtils.reshape(this, shape);
  }

  slice(num: number): Tensor {
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

  toString() {
    return Tensor.tensorFormat.format(this);
  }
}