import TransformOp from "./TransformOp";

export default class ExpOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.exp(a);
  }

}