import Tensor from "../../Tensor";
import TransformOp from "./TransformOp";

export default class PowerOp extends TransformOp {

  private _power: number;
  get power() {
    return this._power;
  }

  constructor(input: Tensor, other: Tensor, result: Tensor, power: number = 1) {
    super(input, other, result);
    this._power = power;
  }

  body(a: number, b?: number): number {
    return Math.pow(a, this.power);
  }

}