import Tensor from "../Tensor";

export default abstract class Operation {

  private readonly _input: Tensor;
  private readonly _other: Tensor;
  private readonly _result: Tensor;

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

  constructor(input: Tensor, other: Tensor, result: Tensor) {
    this._input = input;
    this._other = other;
    this._result = result;
  }

  body(a: number, b?: number): number {
    return a;
  }

  abstract exec(dim?: number): void;

}