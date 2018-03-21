import TransformOp from "./TransformOp";

export default class RoundOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.round(a);
  }

}