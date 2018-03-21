import TransformOp from "./TransformOp";

export default class AtanOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.atan(a);
  }

}