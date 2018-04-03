import Operation from "../Operation";
import Tensor from "../../Tensor";

export default class LinspaceOp extends Operation {

  private readonly _num: number;
  private readonly _result: Tensor;
  private readonly _start: number;
  private readonly _step: number;
  private readonly _stop: number;

  get isSpecial() {
    return true;
  }

  get num(): number {
    return this._num;
  }

  get result() {
    return this._result;
  }

  get start(): number {
    return this._start;
  }

  get step(): number {
    return this._step;
  }

  get stop(): number {
    return this._stop;
  }

  constructor(result: Tensor, start: number = 0, stop: number = 1, num: number = 1) {
    super([], [result]);
    this._start = start;
    this._stop = stop;
    this._num = num;
    this._result = result;
    this._step = num === 1 ? 0 : (stop - start) / (num - 1);
  }

  exec() {
    let result = this.result.data;
    let val = this._start;
    for (let i = 0; i < result.length; i++) {
      result[i] = val;
      val += this._step;
    }
  }

}