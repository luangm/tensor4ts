import ShapeUtils from "./utils/ShapeUtils";

export default class Shape {

  private _length: number;
  private _order: string;
  private _rank: number;
  private _shape: number[];
  private _shapeStrides: number[];
  private _strides: number[];

  get length() {
    return this._length;
  }

  get order() {
    return this._order;
  }

  get rank() {
    return this._rank;
  }

  get shape() {
    return this._shape;
  }

  get strides() {
    return this._strides;
  }

  constructor(shape: number[], strides?: number[], order: string = 'c') {
    this._shape = shape;
    this._rank = shape.length;
    this._shapeStrides = ShapeUtils.getStrides(shape);
    this._strides = strides || this._shapeStrides;
    this._length = ShapeUtils.getLength(shape);
    this._order = order;
  }

  /**
   * This reverses getOffset
   */
  getIndices(offset: number): number[] {
    let indices = new Array(this.rank);
    for (let i = 0; i < this.rank; i++) {
      indices[i] = Math.floor(offset / this._shapeStrides[i]);
      offset %= this._shapeStrides[i];
    }
    return indices;
  }

  getOffset(indices: number[]): number {
    let offset = 0;
    for (let i = 0; i < this.rank; i++) {
      offset += indices[i] * this.strides[i];
    }
    return offset;
  }

  toString() {
    return this._shape;
  }
}