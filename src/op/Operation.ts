import Tensor from "../Tensor";

export default abstract class Operation {

  private _input: Tensor;
  private _other: Tensor;
  private _result: Tensor;

  constructor(input: Tensor, other: Tensor, result: Tensor) {
    this._input = input;
    this._other = other;
    this._result = result;
  }

  get input() {
    return this._input;
  }

  get isSpecial() {
    return false;
  }

  get other() {
    return this._other;
  }

  get result() {
    return this._result;
  }

  abstract body(a: number, b?: number): number;

  abstract exec(dim?: number): void;


}