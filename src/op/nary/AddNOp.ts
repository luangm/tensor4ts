import Tensor from "../../Tensor";
import NaryOp from "./NaryOp";

export default class AddNOp extends NaryOp {

  constructor(list: Tensor[], result: Tensor) {
    super(list, result);
  }

  body(values: number[]): number {
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      sum += values[i];
    }
    return sum;
  }

}