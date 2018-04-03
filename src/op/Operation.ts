import Tensor from "../Tensor";

export default abstract class Operation {

  private readonly _inputs: Tensor[];
  private readonly _results: Tensor[];

  get inputs() {
    return this._inputs;
  }

  get isSpecial() {
    return false;
  }

  get results() {
    return this._results;
  }

  protected constructor(inputs: Tensor[], results: Tensor[]) {
    this._inputs = inputs;
    this._results = results;
  }

  abstract exec(dim?: number | number[]): void;

}