(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-a2bb5948"],{"3c34":function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"app-container documentation-container"},[n("iframe",{staticStyle:{width:"100%",height:"calc(100vh - 100px)"},attrs:{src:t.docSrc}})])},c=[],a=n("7c70"),s="https://depintel.com/documentation/v087_chinese/_build/html/index.html",r="https://depintel.com/documentation/v087/_build/html/usage/learning/silas_ml.html",u={name:"Documentation",components:{DropdownMenu:a["a"]},data:function(){return{}},computed:{language:function(){return this.$store.getters.language},docSrc:function(){return"zh"===this.language?s:r}}},l=u,o=(n("dcb3d"),n("2877")),d=Object(o["a"])(l,i,c,!1,null,"7580718a",null);e["default"]=d.exports},7459:function(t,e,n){"use strict";n("d8f2")},"7c70":function(t,e,n){"use strict";var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"share-dropdown-menu",class:{active:t.isActive}},[n("div",{staticClass:"share-dropdown-menu-wrapper"},[n("span",{staticClass:"share-dropdown-menu-title",on:{click:function(e){return e.target!==e.currentTarget?null:t.clickTitle(e)}}},[t._v(t._s(t.title))]),t._l(t.items,(function(e,i){return n("div",{key:i,staticClass:"share-dropdown-menu-item"},[e.href?n("a",{attrs:{href:e.href,target:"_blank"}},[t._v(t._s(e.title))]):n("span",[t._v(t._s(e.title))])])}))],2)])},c=[],a={props:{items:{type:Array,default:function(){return[]}},title:{type:String,default:"vue"}},data:function(){return{isActive:!1}},methods:{clickTitle:function(){this.isActive=!this.isActive}}},s=a,r=(n("7459"),n("2877")),u=Object(r["a"])(s,i,c,!1,null,null,null);e["a"]=u.exports},"827a":function(t,e,n){},d8f2:function(t,e,n){},dcb3d:function(t,e,n){"use strict";n("827a")}}]);